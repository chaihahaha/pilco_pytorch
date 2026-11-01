# 手推雅可比 vs 反向传播：`gp0d.py` / `gp1d.py` 的矩运算能不能简化？

> 结论先行：**能，而且可以整块删掉——但只能删"导数"那一半，不能删"矩"那一半。**
>
> `gp0d.py` 里 6 个手推 Jacobian（84 行）100% 可以由 autograd 生成，实测误差
> $10^{-11}$；而前向的 $\mathbb{E}[k]$、$\mathbb{E}[kk^\top]$ 一行都不能省，
> 因为**它们是积分，不是导数**，autograd 不积分。
>
> 另外有三个真问题必须在删之前想清楚：对称矩阵上的导数约定、缓存与计算图的冲突、
> 前向模式与反向模式的成本差。下面逐个说。

复现本节所有数字：

```bash
PYTHONPATH=$(cd ..; pwd) python scripts/diagnostics/autograd_vs_analytic_derivatives.py
```

---

## 1. `gp0d.py` 到底算了什么

`gp0d.py` 共 233 行，可以切成四段。前三段是数学，最后一段是"给 MATLAB 的 `vec`
擦屁股"的仪式：

| 行号 | 内容 | 性质 | 能否删 |
| --- | --- | --- | --- |
| 80–107 | 缓存 $A=K+\sigma_n^2I$ 的 Cholesky、$A^{-1}$、$\beta=A^{-1}Y$ | 数值技巧 | 可换成 autograd 内部的求解，但缓存本身有价值 |
| 116–153 | 前向：$\phi_n$、$M$、$V$、$C$ | **积分** | ❌ 不能删 |
| 155–219 | 前向：$\Psi_{ef}$、$\mathrm{Cov}[f_e,f_f]$ | **积分** | ❌ 不能删 |
| 139–152, 173–197, 210–218 | 6 个 Jacobian 的赋值 | **导数** | ✅ 全部可删 |
| 226–233 | `reshape(..., order='F')` 向量化 | 表示法 | ✅ 全部可删 |

**最强的证据是文件对立本身**：仓库里每个文件都有一对

```
gp0.py   149 行   ← 只有前向
gp0d.py  233 行   ← 前向 + 导数（+84 行）
gp1.py   173 行   ← 只有前向
gp1d.py  242 行   ← 前向 + 导数（+69 行）
```

用 `diff` 比对 `gp0.py` 与 `gp0d.py` 的"§2 前向"那一段，差异**只有**
`if compute_derivatives:` 包住的行和一个变量改名（`tiL` → `tL`）。也就是说
**"导数"在代码里是物理可分离的**——原作者当年就是这么维护的，只是选择了手推。

### 1.1 六个 Jacobian 分别是什么

$$
\underbrace{dMdm}_{E\times D},\
\underbrace{dSdm}_{E^2\times D},\
\underbrace{dVdm}_{DE\times D},\
\underbrace{dMds}_{E\times D^2},\
\underbrace{dSds}_{E^2\times D^2},\
\underbrace{dVds}_{DE\times D^2}
$$

它们全是**同一个东西**：$(M,S,V)$ 这个三元组对 $(m,s)$ 的 Jacobian。而 $(M,S,V)$
本身就是 $(m,s)$ 的**显式闭式函数**（§2 那两段代码）。所以

$$
\text{Jacobian}\;\Longleftrightarrow\;\text{对已写下的表达式做链式法则}
$$

链式法则是 autograd 唯一会做的事。**这就是为什么可以整块删掉。**

### 1.2 为什么 PILCO 需要它们

因为 `base/value.py` 用的是**前向模式**的链式传播：

```python
dmOdp  = zeros((D, len(p_flat)))       # dm/dp
dSOdp  = zeros((D*D, len(p_flat)))     # dS/dp
...
dmdp_total = dmdmO @ dmOdp + dmdSO @ dSOdp + dmdp
dSdp_total = dSdmO @ dmOdp + dSdSO @ dSOdp + dSdp
```

每个时间步把 $D\times P$ 和 $D^2\times P$ 的矩阵往前推。这需要**算子对 $(m,S)$ 的
完整 Jacobian**（`dmdmO, dSdmO, dmdSO, dSdSO`），所以 PILCO 必须手推它们。

`pilco_pytorch` 换成**反向模式**：整个时域连成一张图，`score.backward()` 一次拿到
$dJ/dp$，根本不需要算子的 Jacobian。

---

## 2. 为什么导数块可以完全删掉：实测

`scripts/diagnostics/autograd_vs_analytic_derivatives.py` 在一个
$(n,D,E)=(30,3,2)$ 的随机 GP 上做两件事：取 `gp0d(..., compute_derivatives=True)`
的 9 个输出作为"标准答案"，再用 `torch.func.jacfwd` 对 `GpDynamicsOp.moments`
求一次完整 Jacobian，逐块比较。

```
1) forward moments
   M  max|torch - numpy| = 5.8e-11
   S  max|torch - numpy| = 4.5e-11
   V  max|torch - numpy| = 1.7e-11

2) six hand-derived Jacobians, recomputed by autograd
   dMdm   1.7e-11        dSdm  2.5e-11      dVdm  7.4e-11
   dMds   6.3e-11        dSds  3.0e-11      dVds  见 §4.1
```

**五个在 $10^{-11}$ 量级一致**。这个量级正是"同一公式、不同运算顺序"的舍入差，
不是近似误差。

### 2.1 顺带发现：六个里有四个是独立的

手写代码里有两行看起来很"技巧"的赋值：

```python
dMdm = V.T
dVdm = 2 * np.transpose(dMds, (1, 0, 2))
```

它们不是近似，是**恒等式**：

$$
\frac{\partial M}{\partial m} = V^\top,
\qquad
\frac{\partial V}{\partial m} = 2\Big(\frac{\partial M}{\partial S}\Big)^\top .
$$

第一条：$V\equiv S^{-1}\mathrm{Cov}[x,f]$ 的定义恰好就是 $\partial M/\partial m$。
第二条：对高斯，$\partial_{S_{ij}}=\frac12\partial^2_{m_im_j}$，所以
$\partial V/\partial m$ 与 $\partial M/\partial S$ 只差转置和因子 2。

所以真正的独立量只有 `dMds, dSds, dSdm`（三个），`dMdm` 和 `dVdm` 是白送的。
验证脚本里两条恒等式的残差是 $0$ 和 $2.8\times10^{-17}$。

**autograd 不需要知道这两条**，但它会算出同样的数。（如果想让 autograd 也"省"下
这两个张量，就得自己共享子图——这是 §5 要谈的成本问题。）

---

## 3. 为什么前向块一行都不能删：autograd 不积分

这是最容易被误解的一点。有人会想："既然 autograd 这么强，那 $\mathbb{E}[k(x,X_n)]$
能不能也交给它？"

**不能，而且原因是范畴性的**：

> autograd 只在**已有表达式**上做链式法则。它不产生新的闭式表达式。

$\phi_n=\mathbb{E}_{x\sim\mathcal{N}(m,S)}[k(x,X_n)]$ 是一个**积分**。它的闭式解

$$
\phi_n=\sigma_f^2|B|^{-1/2}\exp\!\big(-\tfrac12\delta_n^\top\tilde\Lambda B^{-1}\tilde\Lambda\delta_n\big)
$$

是从"高斯乘高斯还是高斯 + 配方法"推出来的，不是任何东西的导数。你若想绕过它，只有：

1. **蒙特卡洛**：$\phi_n\approx\frac1M\sum_j k(x^{(j)},X_n)$，$x^{(j)}\sim\mathcal{N}(m,S)$。
   这可以求导，但得到的是**估计量的导数**，方差 $O(M^{-1/2})$，而 PILCO 需要的是
   $10^{-12}$ 量级的精确矩。
2. **数值积分**：$D$ 维网格，$O(N^D)$。
3. **重新推导**：就是 §2 那段代码。

所以分工是清楚的：

$$
\boxed{\ \text{积分}\ \to\ \text{人写闭式};\qquad \text{导数}\ \to\ \text{autograd}\ }
$$

这也解释了 `pilco_pytorch` 的规模：`ops/gp.py::moments` 有约 120 行，全都是
$\phi_n,\Psi_{ef},M,\mathrm{Cov},C$ 的组装——**一行导数都没有**。

### 3.1 一个反例：为什么"让 autograd 帮忙"在别处又成立

`gp/gp2d.py` 额外给出 `dMdi, dMdt, dMdX`（对诱导输入、目标值、训练输入的导数）。
这些**同样**是链式法则的产物，所以也整块可删。`pilco_pytorch` 甚至不需要它们，
因为诱导输入在本项目里没有被优化（cart-pole 传的诱导集是空的，`gp1d` 会退回
`gp0d`）。

而 `gp/hypCurb.py` 手推的超参数梯度**也**可删——`pilco_pytorch` 直接用
`nlml().backward()`。这是收益最大的一处，因为超参数梯度手推最繁琐、最容易错。

---

## 4. 三个必须在删之前想清楚的问题

### 4.1 对称矩阵上的导数约定

实测里唯一"对不上"的是 $\partial V/\partial s$：原始差 $6.3\times10^{-3}$。
但：

| 检查 | 结果 |
| --- | --- |
| 对称化之后 $\|\mathrm{sym}(J_{\text{auto}})-\mathrm{sym}(J_{\text{hand}})\|$ | $7.0\times10^{-11}$ |
| 手写块本身是否已对称 | 是（$\|{\rm sym}(J)-J\|=3.5\times10^{-18}$） |
| 沿随机**对称**方向做有限差分，与手写 $J$ 缩并 | 差 $8.1\times10^{-8}$ |
| 同上，与 autograd $J$ 缩并 | 差 $8.1\times10^{-8}$ |

**结论**：两者都是正确的方向导数，只是把导数"摊在 $(j,k)$ 还是 $(k,j)$"不同。
因为 $s$ 是对称的，实际扰动 $\Delta s$ 一定对称，缩并时这个差别**恒等于零**。

**实践含义**：手写代码返回的是"已对称化"的版本，autograd 返回原始版本。若要把两者
逐元素比对（比如写回归测试），必须先对称化，否则会误判为 bug。反过来说，**这也说明
"手推导数"这类代码的正确性判据本身就很微妙**——它是一个约定问题，不是对错问题。

### 4.2 缓存 vs 计算图：超参数梯度怎么办

`gp0d.py` 的 §1 把 $A^{-1}$ 和 $\beta$ **按超参数 $X$ 缓存**起来（`_cache`，用
`np.array_equal(X, oldX)` 判断失效）。这是手工的"记忆化"。

在有 autograd 的世界里，缓存与求导是冲突的：**如果你要对超参数求导，$A^{-1}$
就不能被当成常数**，它必须留在图里（否则梯度会漏掉
$\partial A^{-1}/\partial\text{hyp} = -A^{-1}(\partial A)A^{-1}$ 这一项）。

`pilco_pytorch` 的处理（`GpDynamicsOp._solve_cache`）是：

- 超参数**不需要**梯度时（策略优化阶段）：缓存分解，省掉 $O(n^3)$；
- 超参数**需要**梯度时（GP 训练阶段）：旁路缓存，每次重算 Cholesky 并留在图里。

这不是"更聪明的实现"，而是"把原来手工做的选择显式化了"。原来的 `gp0d` 把这两件事
混在一个函数里，靠调用者传 `compute_derivatives` 来区分；现在靠 `requires_grad` 区分。

> **这就是"图在哪里被切断"这个设计决策**。切断点必须选在"该量真的是常数"的地方。
> `pilco_pytorch` 在 `optimise_policy` 里写 `gp_params.requires_grad_(False)`，
> 语义是"模型在这一阶段是常数"——一个**算法**决策，不是实现细节。

### 4.3 前向模式 vs 反向模式：P 个方向 vs 1 个方向

这是成本账里最关键、也最容易被忽略的一点。

PILCO 的 `value.py` 传播 $\partial m/\partial p\ (D\times P)$ 和
$\partial S/\partial p\ (D^2\times P)$。**每一步的成本是 $O(P)$**（矩阵乘法的列数
是 $P$）。cart-pole 的策略参数 $P=12\times5+12\times1+5=77$。

反向模式相反：一次 `backward()` 得到**全部** $P$ 个梯度，**成本与 $P$ 无关**。

$$
\text{PILCO: } O(P)\ \text{per step}
\qquad\text{vs}\qquad
\text{autograd: } O(1)\ \text{per step}
$$

所以当 $P\gg D$ 时（这里 $77\gg 5$），反向模式不只是"少写代码"，而是**渐近更便宜**。
这也是为什么 `pilco_pytorch` 能在保持精确性的同时把策略优化做快——虽然它的绝对耗时
被 GP 矩的计算主导了（见 `docs/ARCHITECTURE.md` §4b 的 profile）。

**但反向模式也有它贵的地方**：如果你需要的是**完整 Jacobian**（而不是 VJP），
反向模式要跑输出维数那么多次。`plan(with_jacobians=True)` 就是这种情况（要
$J_t=\partial m_{t+1}/\partial m_t$，一个 $D\times D$ 的完整 Jacobian），实测是
普通 `plan` 的 **5.46 倍**。`ExplorationObjective.needs_jacobians` 因此只在
真的要算累积代价方差时才请求它。

---

## 5. 成本账

同一次运行里测的三种策略（以"前向一次"为单位）：

| 方式 | 相对成本 | 产出 |
| --- | --- | --- |
| 手写完整 Jacobian（`compute_derivatives=True`） | $2.1\times$ | 6 个张量全给 |
| `torch.func.jacfwd` 完整 Jacobian | $4.5\text{–}6.4\times$ | 6 个张量全给 |
| `backward()` 一次 VJP | $1.3\text{–}3.0\times$ | 只有标量目标的梯度 |

**读法**：

1. 在"要完整 Jacobian"这件事上，手写代码与 autograd 是**同一量级**（2～6 倍），
   没有数量级差异。autograd 略贵，因为它老实算了 $dSds$ 这种标量目标用不到的块。
2. 在"只要标量梯度"这件事上，VJP 反而比手写"完整 Jacobian"便宜——因为它根本不
   物化那些张量。
3. **绝对耗时不可比**：torch 在 $3\times3$ 矩阵上每次算子调用有固定开销，
   前向就已经比 NumPy 慢一个量级。这是实现常数，不是算法差异。

所以重写的收益排序是：

$$
\text{正确性} \;>\; \text{可维护性} \;>\; \text{速度}
$$

而不是反过来。

---

## 6. `gp1d.py` / `gp2d.py` 的推广

同一个结论逐字成立，只是矩阵变成"诱导点"版本：

- `gp1d.py` 的导数块 69 行，结构与 `gp0d.py` 完全相同（$dMdm,dSdm,\dots$），
  同样 100% 可删。
- `gp2d.py` 多出 `dMdi, dMdt, dMdX`（对诱导输入/目标/训练输入），411 行里约
  一半是导数，同样可删。
- 唯一不能删的仍然是 §155–219 那类"两个核的期望 $\Psi_{ef}$"——它是积分。

**为什么 `pilco_pytorch` 干脆没实现 FITC**：cart-pole 场景传的诱导集是
`np.zeros((50,0,1))`，`size==0`，`gp1d` 会**直接退回** `gp0d`。也就是说这一路的
实验走的全是稠密 GP，FITC 的代码从来没被执行过。重写时把它省掉不是偷懒，是删掉了
一条死代码路径。如果要支持大数据集，按同样的方式加一个 `SparseGpDynamicsOp`
即可——只有前向，导数照旧由 autograd 给。

---

## 7. 建议（如果你要在原仓库上改）

1. **保留** `gp/gp0d.py` 的 §2、§3（前向矩），一个字符都别动——它是标准答案，
   用来验证任何重写。
2. **删除**所有 `if compute_derivatives:` 块，以及 §5 的 `order='F'` 向量化。
   改成"返回 $(M,S,V)$ 的纯函数 + 调用者用 autograd"。
3. **不要**试图用蒙特卡洛替代 $\phi_n,\Psi_{ef}$。精度不够，而且方差会让策略优化
   变成随机搜索。
4. **保留** `compute_derivatives=False` 的路径作为回归基准：任何重写都应该能在
   $10^{-9}$ 内复现它。
5. 如果只是想给现有代码加一个"新算子"（比如新的非线性），**完全不用碰 `gp*d.py`**：
   写一个 `moments(m, s)` 返回三元组的函数即可，导数交给 autograd。

---

## 附：一张图看懂

```
                  ┌─────────────────────────────────────┐
   输入 (m, S) ──▶│  φ_n = E[k(x,X_n)]        ← 积分   │──▶ M
                  │  Ψ_ef = E[k(x,X_n)k(x,X_m)] ← 积分  │──▶ Cov
                  │  C = S⁻¹Cov[x,f]          ← 积分    │──▶ cross
                  └─────────────────────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
        ┌────────────────────────┐    ┌──────────────────────────┐
        │ 6 个 Jacobian          │    │ autograd 一次 backward   │
        │ (dMdm … dVds, 84 行)   │    │ (0 行, 一次 VJP)         │
        │ ← 纯链式法则，可删      │    │ ← 纯链式法则，保留        │
        └────────────────────────┘    └──────────────────────────┘
```
