# pilco_pytorch 数学手册

> 目标读者：已经会写 Python、知道矩阵乘法，但对「高斯过程 / 矩匹配 / 矩阵微分」只有
> 初步印象，希望**读代码时不再被公式卡住**的人。
>
> 本文不证明定理，但会讲清楚每个结论**在算什么、为什么需要它、代码里对应哪一行**。
> 所有公式都用 `pilco_pytorch` 实际的符号与布局书写，不是教科书上另一套记号。

**阅读顺序建议**：§0 看符号约定 → §1–§2 建立"世界是高斯"的世界观 → §3–§4 是全文
最硬的部分（GP 在不确定输入下的预测），务必逐行读 → §5–§6 讲这些公式怎么被组装成
`plan()` → §7 有向探索 → §8 导数 → §9 对照表（可以当字典随时回查）。

---

## 0. 符号与约定

### 0.1 符号表

| 符号 | 含义 | 代码里的名字 |
| --- | --- | --- |
| $D$ | 状态维数 | `spec.state_dim` |
| $D_u$ | 控制维数 | `spec.control_dim` |
| $n$ | 训练数据点数 | `inputs.shape[0]` |
| $E$ | 输出（GP）个数 | `targets.shape[1]` |
| $x\in\mathbb{R}^D$ | **随机**状态向量 | `mean`, `cov` 描述其分布 |
| $m,\ S$ | 高斯分布的均值、协方差 | `mean`, `cov` |
| $u$ | 控制量 | `idx_u` |
| $f(\cdot)$ | 真实动力学（未知） | — |
| $g(\cdot)$ | 任意非线性环节 | `MomentOp` 子类 |
| $k(x,x')$ | 核函数 | `RbfPolicyOp.kernel` |
| $X\in\mathbb{R}^{n\times D}$ | 训练输入（堆成矩阵） | `inputs` |
| $Y\in\mathbb{R}^{n\times E}$ | 训练目标 | `targets` |
| $K$ | Gram 矩阵，$K_{ij}=k(X_i,X_j)$ | `K` |
| $A$ | $K+\sigma_n^2 I$，加噪后的 Gram 矩阵 | `_solve_cache` 内部 |
| $\beta$ | $A^{-1}Y$，**权重** | `beta` |
| $\sigma_f$ | 信号标准差（核的幅度） | `signal` |
| $\sigma_n$ | 观测噪声标准差 | `noise` |
| $\ell_d$ | 第 $d$ 维的长度尺度 | `lengthscales` |
| $\Lambda$ | $\mathrm{diag}(1/\ell_1^2,\dots,1/\ell_D^2)$ | `iell2` |
| $\tilde\Lambda$ | $\mathrm{diag}(1/\ell_1,\dots,1/\ell_D)$ | `iell` |
| $\delta_n$ | $m-X_n$，查询点均值到第 $n$ 个训练点 | `ytilde` |
| $\phi_n$ | $\mathbb{E}[k(x,X_n)]$，**一个核的期望** | `ell` ← 注意不是长度尺度！ |
| $\Psi_{ef}[n,n']$ | $\mathbb{E}[k_e(x,X_n)k_f(x,X_{n'})]$，**两个核的期望** | `L_mat`, `L` |
| $C$ | $S^{-1}\mathrm{Cov}[x,g(x)]$，输入-输出交叉协方差 | `Moments.cross` |
| $c(x)$ | 代价函数 | `cost` |
| $H$ | 规划时域 | `horizon` |
| $\gamma$ | 折扣因子 | `gamma` |
| $\theta$ | 策略参数（中心点、目标值、长度尺度） | `policy.parameters()` |

> ⚠️ **第一个坑**：原代码把 $\mathbb{E}[k(x,X_n)]$ 命名为 `ell`，而长度尺度叫
> `log_lengthscales`。本文用 $\phi_n$ 表示前者、$\ell$ 表示后者，读代码时请记住
> `ell == phi`。

### 0.2 三个贯穿全文的约定

**(1) 一切都是列向量。** $x$ 是 $D\times 1$，$m$ 是 $D\times 1$，$S$ 是 $D\times D$。
代码里 `mean` 是 `(D,)` 形状的一维张量——PyTorch 允许省略尾部的 1，但**数学上它就是
列向量**。协方差矩阵 $S$ 的下标约定是 $S_{ij}=\mathrm{Cov}[x_i,x_j]$。

**(2) 协方差矩阵必须对称半正定（PSD）。** $\mathrm{Cov}[x,x]=S$ 永远满足
$S=S^\top$ 且 $v^\top S v\ge 0$。解析公式攒出来的 $S$ 会有 $10^{-16}$ 量级的
不对称，直接丢进 Cholesky 分解会失败。所以代码里几乎每个矩阵在进入线性代数之前都过
一遍 `symmetrize()`，见 `pilco/gaussian.py`。

**(3) `vec` 采用"列优先"（Fortran / MATLAB）顺序。** 当代码需要把矩阵摊平成向量
（例如把 $\partial S/\partial s$ 写成一个 $E^2\times D^2$ 的 Jacobian）时，用的是
MATLAB 的 `vec`：

$$
\mathrm{vec}(A)_{\,i + (j-1)M} \;=\; A_{ij}
\qquad (A \text{ 是 } M\times N)
$$

即**先走完一列再走下一列**。NumPy 里对应 `A.reshape(-1, order='F')`，PyTorch 里
没有 `order='F'`，`pilco_pytorch` 干脆不再需要它——因为 autograd 直接给出张量形式的
Jacobian（见 §8）。但 `pilco_python` 的 `gp*d.py` 里到处是 `order='F'`，读那些文件
时要知道这一点。

### 0.3 本文与代码的对应关系

```
数学                      代码
──────────────────────────────────────────────────────────
高斯分布                  pilco/gaussian.py        Gaussian
一个非线性环节的矩         pilco/ops/base.py        MomentOp, Moments
把输出拼回大高斯           pilco/ops/base.py        splice()
sin/cos 的矩              pilco/ops/fourier.py     TrigAugmentOp
饱和函数的矩               pilco/ops/fourier.py     SaturationOp
代价的矩                  pilco/ops/cost.py        SaturatingCostOp
GP 在不确定输入下的矩       pilco/ops/gp.py          GpDynamicsOp
RBF 策略的矩               pilco/ops/policy.py      RbfPolicyOp
一步的组装                pilco/trajectory.py      step_augmented()
整条时域                  pilco/trajectory.py      plan()
探索目标                  pilco/objectives.py      ExplorationObjective
```

---

## 1. 高斯分布：我们唯一会用的对象

PILCO 的全部魔法来自一个事实：**高斯分布在两种操作下是封闭的**。整个算法就是反复
使用这两种操作。

### 1.1 为什么是高斯

如果 $x\sim\mathcal{N}(m,S)$，那么：

- 经过**仿射变换** $y = Fx + b$，$y$ 仍然是高斯；
- 给定 $x$ 的一部分坐标，**剩下部分的分布**仍然是高斯。

这两条就是全部。PILCO 的每一次"前进一步"都是这两条的组合：把状态推过策略（一个
非线性函数，我们先近似成"条件均值仿射"）、推过动力学（同理）、再取边缘——每一步都
保持"是高斯"这个性质。

**代价**：真实分布不是高斯，我们强行用高斯近似。§2.4 会说明近似的具体含义。

### 1.2 两条基本法则

**法则 A（仿射推送）。** 若 $x\sim\mathcal{N}(m,S)$，$y=Fx+b$，则

$$
\mathbb{E}[y] = Fm+b, \qquad \mathrm{Cov}[y] = F S F^\top .
$$

代码里到处都是这一条。例如 `AffineOp.moments`（`pilco/ops/base.py`）：

```python
m = A @ mean + self.bias
S = symmetrize(A @ cov @ A.transpose(0, 1))
```

**法则 B（条件化）。** 若

$$
\begin{bmatrix} x_a \\ x_b \end{bmatrix}
\sim \mathcal{N}\!\left(\begin{bmatrix} m_a \\ m_b \end{bmatrix},
\begin{bmatrix} S_{aa} & S_{ab} \\ S_{ba} & S_{bb}\end{bmatrix}\right),
$$

则给定 $x_a = v$ 后

$$
x_b \mid x_a=v \;\sim\; \mathcal{N}\!\big(m_b + S_{ba}S_{aa}^{-1}(v-m_a),\;
S_{bb} - S_{ba}S_{aa}^{-1}S_{ab}\big).
$$

对应 `Gaussian.conditional()`（`pilco/gaussian.py`）。**这是 GP 预测的全部内容**
（§3.3）：GP 回归不是新东西，就是法则 B 套在一个大高斯上。

### 1.3 二次型的期望——后面所有的"闭环公式"都靠它

设 $x\sim\mathcal{N}(m,S)$，$W$ 对称。则

$$
\boxed{\;\mathbb{E}\big[e^{-\frac12 x^\top W x}\big]
= |I + SW|^{-1/2}\,
\exp\!\Big(-\tfrac12 m^\top W (I+SW)^{-1} m\Big)\;}
\tag{1.1}
$$

**怎么读这个公式**（这是本文最重要的公式之一）：

- $|I+SW|^{-1/2}$ 是"方差带来的衰减"：$S=0$（输入确定）时它等于 1，$S$ 越大它越小。
- 指数里 $m^\top W (I+SW)^{-1} m$：如果 $S=0$ 就是普通的 $m^\top W m$，加方差后
  被 $(I+SW)^{-1}$ "稀释"了。
- 注意 $SW$ 一般**不对称**，但 $|I+SW|=|I+W^{1/2}SW^{1/2}|$，后者对称，所以实际
  计算时把 $S$ 收缩成对称形式再算行列式（`logdet_psd_scaled_diag` 干的就是这件事）。

同一件事的更一般形式（带线性项），后面会反复用：

$$
\mathbb{E}\big[e^{-\frac12 x^\top A x + b^\top x}\big]
= |I+SA|^{-1/2}
\exp\!\Big(-\tfrac12 m^\top A m + b^\top m
+\tfrac12 (b-Am)^\top (I+SA)^{-1} S (b-Am)\Big).
\tag{1.2}
$$

(1.1) 是 (1.2) 在 $b=0$ 时的特例。**记不住 (1.2) 也没关系**，只要记住一句话：
*高斯下"指数二次型"的期望仍然是指数二次型，系数需要解一个 $D\times D$ 的线性方程*。

### 1.4 协方差的对称性与"半向量化"

$\mathrm{Cov}[x,x]$ 只有 $D(D+1)/2$ 个独立元素。但解析求导时，PILCO 把 $\partial/\partial S$
当成对 $D^2$ 个元素分别求导（把 $S$ 当作普通矩阵），于是产生 $D^2$ 个"导数"。

这会带来一个**约定问题**，它真实地出现在 `gp0d.py` 里：

- 手写代码返回的 $\partial V/\partial S$ **已经对称化**了（对 $(j,k)$ 和 $(k,j)$ 取平均）；
- autograd 返回的是"把 $S$ 当 $D^2$ 个独立变量"的原始导数，**不对称**。

两者相差一个对称化操作。但如果下游是 $\sum_{jk}\frac{\partial V}{\partial S_{jk}}\,\Delta S_{jk}$
且 $\Delta S$ 本身对称，那么两者给出**完全相同**的数。这就是为什么
`scripts/diagnostics/autograd_vs_analytic_derivatives.py` 里 $\partial V/\partial s$
原始差 $6.3\times10^{-3}$、对称化后差 $7\times10^{-11}$。详见 §8.5。

### 1.5 数值安全：为什么要 Cholesky

需要算 $|A|$、$A^{-1}b$、$\log|A|$ 时，**永远不要**先求逆再乘，也**永远不要**用
`det`。正确做法是 Cholesky 分解 $A=LL^\top$：

$$
|A| = \prod_i L_{ii}^2, \qquad
\log|A| = 2\sum_i \log L_{ii}, \qquad
A^{-1}b = L^{-\top}\!\big(L^{-1}b\big).
$$

`pilco/gaussian.py` 里的 `safe_cholesky` 还会在失败时**逐步加抖动**重试
（$10^{-9},10^{-8},\dots$），最后退化为特征值截断投影（`nearest_psd`）。原因很实际：
解析矩匹配攒出的矩阵经常"理论上半正定、数值上不定"。§4.7 有一个真实例子。

---

## 2. PILCO 到底在优化什么

### 2.1 被控系统

离散时间系统

$$
x_{t+1} = f(x_t, u_t) + \varepsilon_t, \qquad \varepsilon_t\sim\mathcal{N}(0,\Sigma_\varepsilon),
\qquad x_0\sim\mathcal{N}(m_0,S_0).
$$

控制由确定性策略给出：$u_t = \pi(x_t;\theta)$。在 cart-pole 里 $\pi$ 是一个 RBF 网络
（§5.7），$\theta$ 是它的中心点、目标值和长度尺度。

**注意**：$f$ 是未知的。我们只有一张数据表 $\{(x_t,u_t)\to x_{t+1}-x_t\}$。
PILCO 用 GP 从数据里学 $f$（§3），然后**在学到的模型里规划**，再把规划出的策略拿到
真实系统上跑，收集新数据，重训模型。这就是整个闭环。

### 2.2 目标函数

$$
J(\theta) = \sum_{t=0}^{H-1}\gamma^{t+1}\,\mathbb{E}\big[c(x_t)\big],
\qquad x_{t+1}=f(x_t,\pi(x_t;\theta)).
$$

$\mathbb{E}[\cdot]$ 是对**初始状态的不确定性 + 过程噪声**取的期望。PILCO 的核心
主张是：*把这个期望的传播做成闭式的，就能对 $J$ 求梯度*。

### 2.3 为什么不能直接求导

要算 $\mathbb{E}[c(x_1)]$ 其中 $x_1 = f(x_0,\pi(x_0))$ 且 $x_0\sim\mathcal{N}(m_0,S_0)$，
你面对的是

$$
\int c\big(f(x,\pi(x))\big)\,\mathcal{N}(x;m_0,S_0)\,dx .
$$

对一般的 $f$（cart-pole 的动力学里有 $\sin$、$\cos$、分母上的 $4(m_c+m_p)-3m_p\cos^2\theta$）
**这个积分没有闭式解**。而且随着 $t$ 增大，$x_t$ 的分布会越来越不像高斯——这是
PILCO 方法的根本局限，也是它在长时域上会失败的原因（见 `docs/ARCHITECTURE.md`）。

### 2.4 PILCO 的核心假设：**把分布近似成高斯**

PILCO 不去算上面那个积分，而是做两件事：

1. **假设 $x_t$ 一直是高斯**：只保留 $m_t, S_t$ 两个量。
2. **假设每个非线性环节的输出也是高斯**，其均值与协方差由"矩匹配"给出。

于是"推进一步"变成：

$$
(m_t,S_t) \;\xrightarrow{\ \text{策略}\ }\; (m_u,S_u)
\;\xrightarrow{\ \text{饱和}\ }\; (m_{u'},S_{u'})
\;\xrightarrow{\ \text{GP 动力学}\ }\; (m_{t+1},S_{t+1}).
$$

每一步都只是"输入一个高斯，输出一个高斯"的函数。**这就是 `MomentOp` 的全部内容**：

```python
class MomentOp(nn.Module):
    def moments(self, mean, cov) -> Moments:   # (m, S) -> (mean, cov, cross)
        ...
```

### 2.5 矩匹配到底"匹配"了什么

给定真实的（非高斯）分布 $p$，我们想找最接近的高斯 $q=\mathcal{N}(\mu,\Sigma)$。
"最接近"的判据是最小化 KL 散度：

$$
\mathrm{KL}\big(q\,\|\,p\big) = \int q(x)\log\frac{q(x)}{p(x)}dx .
$$

可以证明（本文不证）：在全体高斯分布里最小化 $\mathrm{KL}(q\|p)$，得到的
$q$ 恰好满足

$$
\mu = \mathbb{E}_p[x], \qquad \Sigma = \mathrm{Cov}_p[x].
$$

**这就是"矩匹配"这个名字的由来**：匹配一阶矩和二阶矩。所以当我们写
$\mathbb{E}[g(x)]$、$\mathrm{Cov}[g(x)]$ 的闭式公式时，我们算的就是这两个积分，
没有别的玄机。

### 2.6 代价函数

cart-pole 用的代价是（`pilco/ops/cost.py`）

$$
c(x) = 1 - \exp\!\Big(-\frac{1}{2w^2}\,d(x)^2\Big),
\qquad
d(x)^2 = (x_{\text{cart}} + \ell\sin\theta)^2 + \ell^2(\cos\theta+1)^2 .
$$

**为什么长这样**：$d^2$ 是"摆杆末端到目标点 $(0,\ell)$ 的距离平方"。

- 摆杆竖直向上（$\theta=\pi$）且小车在原点时 $d=0$，代价 $0$；
- 摆杆自然下垂（$\theta=0$）时 $d=2\ell$，代价 $\approx 1$。

指数形式的好处是**它的期望有闭式解**（用公式 1.1）——如果写成 $d^2$ 本身，期望虽然
也闭式，但失去"$d$ 小时代价近似二次型"的性质。$w$ 控制"多近才算成功"。

---

## 3. 高斯过程

### 3.1 直觉

一个 GP 是"函数的分布"。写下

$$
f \sim \mathcal{GP}\big(0,\,k(\cdot,\cdot)\big)
$$

的意思是：随便挑有限个输入 $x_1,\dots,x_n$，函数值向量
$\big(f(x_1),\dots,f(x_n)\big)$ 服从一个**联合高斯**，其均值为 0，协方差为

$$
\mathrm{Cov}\big[f(x_i),f(x_j)\big] = k(x_i,x_j).
$$

核函数 $k$ 决定了"函数有多光滑、变化有多快"。**GP 没有给出一个具体函数，它给的是
"哪些形状更可能"**。

### 3.2 平方指数核与它的三个旋钮

PILCO 用的核是 ARD（自动相关确定）平方指数核：

$$
k(x,x') = \sigma_f^2\exp\!\left(-\frac12\sum_{d=1}^{D}\frac{(x_d-x'_d)^2}{\ell_d^2}\right).
\tag{3.1}
$$

三个超参数各管一件事：

| 参数 | 代码 | 作用 |
| --- | --- | --- |
| $\sigma_f$ | `log_signal` | 函数的**幅度**（"值能差多远"） |
| $\ell_d$ | `log_lengthscales` | 第 $d$ 维的**相关长度**（"多远之外就无关了"） |
| $\sigma_n$ | `log_noise` | 观测噪声（"数据有多脏"） |

写成矩阵形式，令 $\Lambda=\mathrm{diag}(1/\ell_1^2,\dots,1/\ell_D^2)$，则

$$
k(x,x') = \sigma_f^2\exp\!\Big(-\tfrac12 (x-x')^\top \Lambda (x-x')\Big).
$$

这个形式是后面**所有闭式积分存在的根本原因**：核是高斯形状，高斯乘高斯还是高斯。

### 3.3 条件化 = 高斯条件化

给定数据 $\{(X_i,y_i)\}_{i=1}^n$，假设 $y_i = f(X_i)+\varepsilon_i$，
$\varepsilon_i\sim\mathcal{N}(0,\sigma_n^2)$。那么

$$
\begin{bmatrix} Y \\ f(x_*) \end{bmatrix}
\sim \mathcal{N}\!\left(
\begin{bmatrix} 0 \\ 0\end{bmatrix},
\begin{bmatrix}
K+\sigma_n^2 I & k_* \\
k_*^\top & k(x_*,x_*)
\end{bmatrix}\right),
$$

其中 $k_* = [k(x_*,X_1),\dots,k(x_*,X_n)]^\top$。套用法则 B：

$$
\mathbb{E}[f(x_*)] = k_*^\top \underbrace{(K+\sigma_n^2 I)^{-1}Y}_{=:\ \beta},
\qquad
\mathrm{Var}[f(x_*)] = k(x_*,x_*) - k_*^\top A^{-1}k_* .
\tag{3.2}
$$

**所以 GP 回归就是一个 $n$ 维的高斯条件化，没有更多。** 代码里 $\beta$ 被缓存起来
（`_solve_cache`），因为它在每个查询点都相同——这是把 $O(n^3)$ 的分解开销摊掉的
关键。

把 (3.2) 的均值写成"核值的线性组合"：

$$
\boxed{\;f(x_*) = \sum_{n=1}^n \beta_n\, k(x_*,X_n)\;}
\tag{3.3}
$$

**这个式子是 §4 的钥匙**：$f$ 对"核值"是**线性**的，而对 $x_*$ 是非线性的。求
$\mathbb{E}[f(x)]$ 时，线性性让我们可以把期望推进求和号里面，只需要算
$\mathbb{E}[k(x,X_n)]$。

### 3.4 边缘似然：怎么选超参数

超参数不靠交叉验证，靠最大化**边缘似然** $p(Y\mid X,\text{hyp})$。对高斯来说

$$
\log p(Y\mid X) = -\frac12 Y^\top A^{-1}Y - \frac12\log|A| - \frac{n}{2}\log 2\pi .
\tag{3.4}
$$

（代码里最小化它的相反数 `nlml`。）三项的意义：

- $-\frac12Y^\top A^{-1}Y$：**拟合程度**。模型预测得准，这一项大。
- $-\frac12\log|A|$：**复杂度惩罚**。$A$ 的行列式大（长度尺度小、噪声大）时这一项小。
- 最后一项是常数。

两项拉扯的结果就是"既解释数据又不过度自信"。PILCO 还额外加了一个"curb"惩罚项
（`curbed_nlml`），把长度尺度往输入的标准差上拉——因为在小数据集上无约束的
极大似然会把噪声压到接近 0，模型过度自信，规划器就不再纠错了。这个现象的实测见
`docs/ARCHITECTURE.md`。

### 3.5 多输出：$E$ 个独立的 GP

动力学有 $D$ 个输出（cart-pole 是 4 个）。PILCO 为每个输出训练**一个独立的 GP**，
它们共享输入 $X$ 但各有自己的长度尺度、信号和噪声：

$$
\beta_e = (K_e+\sigma_{n,e}^2 I)^{-1}Y_{:,e},
\qquad
K_e[i,j]=\sigma_{f,e}^2\exp\!\big(-\tfrac12 (X_i-X_j)^\top\Lambda_e(X_i-X_j)\big).
$$

**关键点**：这 $E$ 个 GP 在给定输入时是**条件独立**的。它们之间的相关性**完全**来自
"我们不知道输入是什么"。这一点决定了 §4.5 里协方差公式的结构（只有 $e=f$ 时才有
$\mathrm{tr}$ 项）。

---

## 4. 不确定输入下的 GP 预测（全文最硬的一节）

### 4.1 问题

规划时我们不知道 $x$，只知道 $x\sim\mathcal{N}(m,S)$。要求

$$
\mathbb{E}[f_e(x)], \qquad \mathrm{Cov}[f_e(x),f_f(x)], \qquad
\mathrm{Cov}[x,f_e(x)] .
$$

第三个量 $C$ 是 PILCO 特有的，§5.2 会解释为什么必须要它。

### 4.2 关键观察

由 (3.3)，$f_e(x)=\sum_n\beta_{e,n}k_e(x,X_n)$ 对核值是线性的，所以

$$
\mathbb{E}[f_e(x)] = \sum_n \beta_{e,n}\,\underbrace{\mathbb{E}[k_e(x,X_n)]}_{=:\ \phi_n}.
$$

于是**问题从"对任意函数求期望"降级成"对核函数求期望"**。而核是高斯形状，所以这个
期望有闭式解。同理，方差需要

$$
\mathrm{Cov}[f_e(x),f_f(x)] = \sum_{n,n'}\beta_{e,n}\beta_{f,n'}
\Big(\underbrace{\mathbb{E}[k_e(x,X_n)k_f(x,X_{n'})]}_{=:\ \Psi_{ef}[n,n']}
- \phi_n\phi_{n'}\Big).
$$

所以我们只需要算两个积分：$\phi_n$（一个核）和 $\Psi_{ef}$（两个核的乘积）。

### 4.3 一个核的期望 $\phi_n$

要算

$$
\phi_n = \mathbb{E}_{x\sim\mathcal{N}(m,S)}\Big[\sigma_f^2 e^{-\frac12 (x-X_n)^\top\Lambda(x-X_n)}\Big].
$$

令 $\delta_n = m - X_n$（代码里 `ytilde = (mean - inputs) * iell`，注意它多乘了一个
$\tilde\Lambda$，见下），$y=x-m$，则 $x-X_n = y+\delta_n$，指数变成

$$
-\tfrac12 y^\top\Lambda y - \delta_n^\top\Lambda y - \tfrac12\delta_n^\top\Lambda\delta_n .
$$

套 (1.2)（取 $A=\Lambda$，$b=-\Lambda\delta_n$），化简得

$$
\boxed{\;\phi_n = \sigma_f^2\,|B|^{-1/2}
\exp\!\Big(-\tfrac12\,\delta_n^\top \tilde\Lambda B^{-1}\tilde\Lambda\,\delta_n\Big)\;}
\qquad
B := I + \tilde\Lambda S\tilde\Lambda
\tag{4.1}
$$

其中 $\tilde\Lambda=\mathrm{diag}(1/\ell_d)$ 是 $\Lambda$ 的"平方根"。

**为什么写成 $B$ 而不是 $I+S\Lambda$**：$B$ 是**对称**的，可以直接 Cholesky。
$|B|=|I+S\Lambda|$（两者特征值相同），所以行列式不变。代码里：

```python
B = eye + (iell[:, None] * iell[None, :]) * cov      # B = I + Λ̃ S Λ̃
ytilde = (mean[None, :] - inputs) * iell             # δ_n Λ̃  （行向量形式）
sol = solve(B, ytilde.T).T                           # B^{-1} Λ̃ δ_n
phi = signal**2 * exp(-0.5*logdet(B) - 0.5*(ytilde*sol).sum(-1))
```

`(ytilde*sol).sum(-1)` 就是 $\delta_n^\top\tilde\Lambda B^{-1}\tilde\Lambda\delta_n$。

**直观**：$S\to 0$ 时 $B\to I$，$\phi_n\to\sigma_f^2\exp(-\frac12\delta_n^\top\Lambda\delta_n)$，
退化成"在均值处算核值"。$S$ 越大，$\phi_n$ 越小——因为核在小范围内剧烈变化，
平均下来被抹平了。

### 4.4 两个核的期望 $\Psi_{ef}$

要算

$$
\Psi_{ef}[n,n'] = \mathbb{E}\Big[\sigma_{f,e}^2 e^{-\frac12(x-X_n)^\top\Lambda_e(x-X_n)}
\cdot \sigma_{f,f}^2 e^{-\frac12(x-X_{n'})^\top\Lambda_f(x-X_{n'})}\Big].
$$

**两个高斯的乘积还是高斯**：把两个指数合并，得到

$$
-\tfrac12 x^\top A x + b^\top x + c,
\qquad
A = \Lambda_e+\Lambda_f,\quad
b = \Lambda_e X_n + \Lambda_f X_{n'}.
$$

再套 (1.2)，并用配方

$$
-\tfrac12 m^\top A m + b^\top m + c
= -\tfrac12(m-X_n)^\top\Lambda_e(m-X_n) - \tfrac12(m-X_{n'})^\top\Lambda_f(m-X_{n'})
=: k_n + k_{n'},
$$

得到

$$
\boxed{\;\Psi_{ef}[n,n'] = \sigma_{f,e}^2\sigma_{f,f}^2\,|I+A S|^{-1/2}
\exp\!\Big(k_n + k_{n'} + \tfrac12\, d^\top G\, d\Big)\;}
\tag{4.2}
$$

其中

$$
d = \Lambda_e(X_n-m) + \Lambda_f(X_{n'}-m),
\qquad
G = (I+SA)^{-1}S = (A^{-1}+S)^{-1}.
$$

**代码里这一行长什么样**（`pilco/ops/gp.py`）：

```python
R = s @ diag(iell2_e + iell2_f) + I          # R = I + S(Λ_e + Λ_f) = I + S A
t = 1/sqrt(det R)                            # = |I+A S|^{-1/2}
L = exp(k_n + k_n' + maha(ii, -ij, solve(R, s)/2))
```

- `ii = (X_n - m) Λ_e`，`ij = (X_{n'} - m) Λ_f`，所以 `ii - (-ij) = d`。
- `maha(a, b, M) = (a-b)^\top M (a-b)`，取 $M=R^{-1}S/2$，得到
  $\tfrac12 d^\top R^{-1}S\, d = \tfrac12 d^\top G d$。✅ 完全一致。

> **这就是"为什么代码里那一堆 `maha` 调用其实是高斯积分"**。看到 `maha(a,b,M)` 就
> 想"这是在算 $\exp$ 里的二次型"，看到 `solve(R,s)` 就想"这是在算 $(I+SA)^{-1}S$"。

### 4.5 拼出均值与协方差

**均值**：$\;M_e = \sum_n\beta_{e,n}\phi_n$。代码 `M = (beta * ell).sum(0)`。

**协方差**：把 $\mathrm{Cov}[f_e,f_f] = \mathbb{E}[f_ef_f]-\mathbb{E}[f_e]\mathbb{E}[f_f]$
展开，得到三项：

$$
\boxed{
\mathrm{Cov}[f_e(x),f_f(x)]
= \underbrace{\delta_{ef}\sigma_{f,e}^2}_{\text{(a) 先验方差}}
- \underbrace{\delta_{ef}\,\mathrm{tr}\!\big(A_e^{-1}\Psi_{ee}\big)}_{\text{(b) 数据带来的信息}}
+ \underbrace{\beta_e^\top\big(\Psi_{ef}-\phi_e\phi_f^\top\big)\beta_f}_{\text{(c) 输入不确定性}}
}
\tag{4.3}
$$

三项的意义：

- **(a)** 没有数据时，GP 的先验方差就是 $\sigma_f^2$。
- **(b)** 观测到数据后，不确定性减少。$\mathrm{tr}(A^{-1}\Psi)$ 是"数据点覆盖了多少"。
  只有 $e=f$ 时有这一项，因为不同输出的噪声是独立的。
- **(c)** 我们对输入 $x$ 不确定，这**增加**了输出的不确定性。注意它是"减掉均值乘积"
  的形式，所以可正可负，但总体上当 $S$ 增大时它增大。

> **为什么 (b) 只有 $e=f$**：给定 $x$ 时 $f_e$ 与 $f_f$ 条件独立
> （$\mathrm{Cov}[f_e,f_f\mid x]=0$，$e\ne f$），所以
> $\mathrm{Cov}[f_e,f_f]=\mathbb{E}_x[\mathrm{Cov}[f_e,f_f|x]]+\mathrm{Cov}_x[\mathbb{E}[f_e|x],\mathbb{E}[f_f|x]]$。
> 第一项为 0，第二项就是 (c)。$e=f$ 时第一项不为 0，产生 (b)。

最后还要加上"函数的不确定性"与"噪声"的区分：**预测方差里不含观测噪声**，
`pilco_pytorch` 的 `moments` 返回的是 $\mathrm{Cov}[f(x)]$，不含 $\sigma_n^2$。

### 4.6 交叉协方差 $C=S^{-1}\mathrm{Cov}[x,f(x)]$

PILCO 还需要第三个量：

$$
C := S^{-1}\,\mathrm{Cov}\big[x,\,f_e(x)\big] \in\mathbb{R}^{D}.
$$

**为什么要除以 $S$**：见 §5.2，这样 $\mathrm{Cov}[x,f]=S\,C$ 的形式让"拼接"变成一次
矩阵乘法。

**怎么算**：用 Stein 引理（对高斯成立：$\mathrm{Cov}[x_i, h(x)]=\sum_j S_{ij}\mathbb{E}[\partial h/\partial x_j]$）。
对 $f_e(x)=\sum_n\beta_{e,n}k_e(x,X_n)$：

$$
\mathrm{Cov}[x,f_e(x)] = \sum_n\beta_{e,n}\,\sigma_{f,e}^2\,\phi_n\,\big(-\Lambda_e(x-X_n)\big)\Big|_{x=m}\cdot S
$$

整理成 $S C$ 的形式，得到

$$
\boxed{\;C = -\,\tilde\Lambda_e\,B^{-1}\,\tilde\Lambda_e
\sum_n \beta_{e,n}\,\phi_n\,\delta_n\;}
\tag{4.4}
$$

其中 $B=I+\tilde\Lambda_eS\tilde\Lambda_e$，$\delta_n=m-X_n$。代码：

```python
g = sum_n beta_n * phi_n * (m - X_n)          # 向量
C = -iell * solve(B, iell * g)                 # -Λ̃ B^{-1} Λ̃ g
```

**直观**：$C$ 是"输入的每一维与输出之间的相关性"。如果 $\phi_n$ 在某个方向上变化快，
$C$ 就大。

### 4.7 一个真实的数值陷阱：拼出来的矩阵可以不是 PSD

[`../pilco_python/docs/BUGS.md`](../pilco_python/docs/BUGS.md) 的 B4 记录了一个
实测结果（本仓库的端到端测试也复现了）：

> 解析公式算出的**边缘**协方差与"由线性化交叉协方差拼出的联合分布"**互相矛盾**，
> 相差 28%–55%，导致拼出来的增广协方差矩阵最小特征值是 $-0.79$（应当 $\ge 0$）。

**为什么会这样**：矩匹配是逐环节做的。每个环节分别匹配自己的 $(m,S)$，但
"环节 A 的输出与环节 B 的输入之间的协方差"是用**线性化**（仿射条件均值假设）算的。
两个近似不保证相容。于是拼出来的大矩阵可能不是 PSD。

这是**方法本身的局限**，不是代码 bug。实践中的处理是：所有 Cholesky 前都加抖动
（`safe_cholesky`），并接受这一点点不一致。读代码时看到 `symmetrize` 和 `jitter`
不要以为是洁癖，它们是必需的。

---

## 5. 算子代数：矩匹配如何被组装

### 5.1 `MomentOp`：一个统一的接口

每个非线性环节都实现同一个函数：

$$
(m,S)\ \longmapsto\ \big(\underbrace{\mathbb{E}[g(x)]}_{\text{mean}},\
\underbrace{\mathrm{Cov}[g(x)]}_{\text{cov}},\
\underbrace{S^{-1}\mathrm{Cov}[x,g(x)]}_{\text{cross}}\big).
$$

```python
@dataclass
class Moments:
    mean: torch.Tensor    # (n_out,)
    cov:  torch.Tensor    # (n_out, n_out)
    cross: torch.Tensor   # (n_in, n_out)  == C
```

**为什么只要这三个**：因为只要知道这三个，$g(x)$ 就能被"无缝"地插进任何更大的高斯
（下一节）。PILCO 的全部算子——三角、饱和、GP、代价、策略——都是这一个接口的实例。

### 5.2 `splice`：为什么 $C$ 是关键的第三个数

假设我们已经有一个增广高斯 $z\sim\mathcal{N}(\mu,\Sigma)$，现在要加进一个新的块
$g= g(x_{\text{in}})$。我们希望新的联合分布仍然是高斯：

$$
\begin{bmatrix} z \\ g\end{bmatrix}
\sim\mathcal{N}\!\left(\begin{bmatrix}\mu\\ \mu_g\end{bmatrix},
\begin{bmatrix}\Sigma & \Sigma_{zg}\\ \Sigma_{zg}^\top & \Sigma_g\end{bmatrix}\right).
$$

$\mu_g$ 和 $\Sigma_g$ 由算子给出。$\Sigma_{zg}$ 呢？

**假设**：$g$ 在给定 $x_{\text{in}}$ 时的**条件均值是仿射的**：

$$
\mathbb{E}[g\mid x_{\text{in}}] \approx \mu_g + C^\top S_{\text{in}}^{-1}(x_{\text{in}}-m_{\text{in}}).
$$

在这个假设下，对任意 $z_j$：

$$
\mathrm{Cov}[z_j,g] = \mathrm{Cov}\big[z_j,\ \mathbb{E}[g\mid x_{\text{in}}]\big]
= \mathrm{Cov}[z_j,x_{\text{in}}]\,S_{\text{in}}^{-1}\,\mathrm{Cov}[x_{\text{in}},g]
= \Sigma[j,\text{in}]\,C .
$$

所以拼接就是一行矩阵乘法（`pilco/ops/base.py`）：

```python
proj = cov[:, in_idx] @ res.cross        # Cov[z_j, g] = Σ[j, in] C
new_mean[out_idx] = res.mean             # 填均值
new_cov[:, out_idx] = proj               # 填交叉列
new_cov[out_idx, :] = proj.T             # 填交叉行
new_cov[out_idx, out_idx] = res.cov      # 对角块以算子为准
```

**这就是整个 PILCO 的"一步"**。原版 `base/propagated.py` 用约 140 行 Fortran 序的
索引运算做同一件事，`splice()` 是它的 10 行通用版本。

> **理解要点**：`cross` 之所以要写成 $S^{-1}\mathrm{Cov}[x,g]$ 而不是直接给
> $\mathrm{Cov}[x,g]$，就是为了上面那个 $\Sigma[j,\text{in}]\,C$ 里 $S^{-1}$ 与
> $\Sigma[j,\text{in}]$ 的"部分抵消"。这是 PILCO 记号里最不直观、但最省事的一处设计。

### 5.3 增广布局

一步之内要加 5 个块，所以先开一个足够大的"槽位"向量，再依次填：

$$
\big[\ \underbrace{x}_{D}\ \big|\ \underbrace{\sin,\cos}_{2A}\ \big|\
\underbrace{u_{\text{raw}}}_{D_u}\ \big|\ \underbrace{u}_{D_u}\ \big|\
\underbrace{f(x,u)}_{D}\ \big]
$$

| 槽位 | 代码 | 谁填它 |
| --- | --- | --- |
| state | `idx_state` | 初始分布 |
| trig | `idx_trig` | `TrigAugmentOp`（§5.4） |
| $u_{\text{raw}}$ | `idx_u_raw` | 策略 `RbfPolicyOp`（§5.7） |
| $u$ | `idx_u` | `SaturationOp`（§5.5） |
| dyn | `idx_dyn` | `GpDynamicsOp`（§4） |

最后一步是**投影**：下一个状态 $=$ 当前状态 $+$ 状态增量（对差分输出的坐标）。

$$
x_{t+1} = P\,z, \qquad P = \big[\,I_{\text{diff}}\ \big|\ 0\ \big|\ 0\ \big|\ 0\ \big|\ I_D\,\big]
$$

`P[k,k]=1` 当且仅当第 $k$ 个输出是"差分"（`difference_mask`）。这就是
`StateSpec.projection()` 与 `step_augmented` 末尾的 $P@\dots$。

> **为什么要有 $u_{\text{raw}}$ 和 $u$ 两个槽**：策略输出的是无界控制，饱和之后才是
> 真正施加的控制。两个槽都留着，是因为它们在同一个高斯里，天然携带相关性。

### 5.4 三角算子：$\sin/\cos$ 的矩

对 $a\sim\mathcal{N}(\mu,\sigma^2)$：

$$
\mathbb{E}[\sin a] = e^{-\sigma^2/2}\sin\mu, \qquad
\mathbb{E}[\cos a] = e^{-\sigma^2/2}\cos\mu .
\tag{5.1}
$$

**这个 $e^{-\sigma^2/2}$ 是全部直觉所在**：角度不确定时，$\sin$ 的期望被"拉向 0"。
$\sigma$ 越大，拉得越狠。确定性角度（$\sigma=0$）时就是普通的 $\sin\mu$。

二阶矩（对两个可能不同的角 $u,v$，协方差为 $c$）：

$$
\begin{aligned}
\mathbb{E}[\sin u\sin v] &= \tfrac12\big[\cos(\mu_u-\mu_v)e^{-\frac12(a+b-2c)}
                            - \cos(\mu_u+\mu_v)e^{-\frac12(a+b+2c)}\big],\\
\mathbb{E}[\cos u\cos v] &= \tfrac12\big[\cos(\mu_u-\mu_v)e^{-\frac12(a+b-2c)}
                            + \cos(\mu_u+\mu_v)e^{-\frac12(a+b+2c)}\big],\\
\mathbb{E}[\sin u\cos v] &= \tfrac12\big[\sin(\mu_u+\mu_v)e^{-\frac12(a+b+2c)}
                            + \sin(\mu_u-\mu_v)e^{-\frac12(a+b-2c)}\big].
\end{aligned}
\tag{5.2}
$$

**怎么记**：$\mu_u-\mu_v$ 那一项配 "$\mathrm{Var}(u-v)$"（$a+b-2c$），$\mu_u+\mu_v$ 那
一项配 "$\mathrm{Var}(u+v)$"（$a+b+2c$）。前者是"两个角一起转"，后者是"反向转"。

**交叉协方差**（用 Stein 引理）：

$$
\mathrm{Cov}[x_j,\sin a] = S_{ja}e^{-\sigma^2/2}\cos\mu, \qquad
\mathrm{Cov}[x_j,\cos a] = -S_{ja}e^{-\sigma^2/2}\sin\mu .
\tag{5.3}
$$

**注意 (5.3) 的优美之处**：$\mathrm{Cov}[x,\sin a]=S_{:,a}\cdot e^{-\sigma^2/2}\cos\mu$，
即它完全落在 $S$ 的第 $a$ 列方向上。除以 $S$ 之后得到 $C$，于是 $C$ 只有第 $a$ 行
非零、值为 $e^{-\sigma^2/2}\cos\mu$。代码正是这样写的（`out_cross` 只在
`(idx, arange(K))` 上放 `coeff`）。

频率不为 1 时（例如 $\sin 3a$），把 $a$ 换成 $3a$ 即可：$\mu\to3\mu$，
$\sigma^2\to9\sigma^2$。代码用 `Harmonic(index, freq, kind, scale)` 表达这件事，
`_harmonics_moments` 里先算 `mu = freq*mean[idx]`、`vv = freq**2*diag(cov)[idx]`。

### 5.5 饱和算子

PILCO 的控制限幅不是 `clip`，而是一个光滑的奇函数：

$$
\mathrm{sat}(u) = u_{\max}\cdot\frac{9\sin u + \sin 3u}{8}.
\tag{5.4}
$$

- 在 $u=0$ 处线性化：$\frac{d}{du}\frac{9\sin u+\sin 3u}{8}\big|_{u=0}=\frac{9+3}{8}=1.5$，
  所以小信号增益是 $1.5\,u_{\max}$。
- 在 $u=\pi/2$ 处取到最大值 $1$（因为 $\frac{9\cdot1+\sin\frac{3\pi}{2}}{8}=\frac{9-1}{8}=1$），
  之后回落。所以它把 $|u_{\text{raw}}|\lesssim\pi/2$ 映射到 $\pm u_{\max}$ 内的光滑饱和。
- **为什么不用 `clip`**：`clip` 的分段线性让矩匹配要么不闭式、要么需要处理拐点。
  而 (5.4) 是**有限的 Fourier 级数**，它的矩可以直接用 §5.4 的公式拼出来。

代码就是"两个 Harmonic（频率 1 和 3，都带 $\sin$）+ 一个线性 readout 矩阵
$[\frac98u_{\max},\frac18u_{\max}]$"。这正好复现了原版 `gSat.py` 里
"把 $[m;3m]$ 和 $\mathrm{blkdiag}(S,9S)$ 堆起来"的技巧，但用不着手推导数。

### 5.6 代价算子

$$
c(x) = 1 - e^{-\frac12 d^\top W d}, \qquad d = x-z,\quad W = Q/w^2 .
$$

用 (1.1)（取 $x\equiv$ 状态、$W$ 为权重）：

$$
e_1 := \mathbb{E}\big[e^{-\frac12 d^\top Wd}\big]
= |I+SW|^{-1/2}\exp\!\Big(-\tfrac12 d_m^\top W(I+SW)^{-1}d_m\Big),
\qquad d_m = m-z,
$$

于是

$$
\boxed{\mathbb{E}[c] = 1-e_1, \qquad
\mathrm{Var}[c] = e_2 - e_1^2, \qquad
e_2 := \mathbb{E}\big[e^{-d^\top Wd}\big] = |I+2SW|^{-1/2}\exp(\cdot)\;}
\tag{5.5}
$$

**为什么 $\mathrm{Var}=e_2-e_1^2$ 而不是 $\mathbb{E}[c^2]-\mathbb{E}[c]^2$**：
$\mathbb{E}[c^2]=\mathbb{E}[(1-e^{-a})^2] = 1-2e_1+e_2$，而
$\mathbb{E}[c]^2=(1-e_1)^2=1-2e_1+e_1^2$，相减正好是 $e_2-e_1^2$。代码里
`variance = clamp(e2 - e1**2, min=0)`。

**跨时刻的代价协方差**（§7.2 要用）：

$$
\mathbb{E}[c(x_1)c(x_2)] = 1 - e_1(x_1) - e_1(x_2) + \mathbb{E}\big[e^{-a_1-a_2}\big],
$$

最后一项是把 $[x_1;x_2]\sim\mathcal{N}$ 拼成一个 $2D$ 维高斯、用块对角
$W_2=\mathrm{blkdiag}(W,W)$ 再套一次 (1.1)。这就是 `cross_moment()`。

### 5.7 策略算子

策略是一个 RBF 网络（PILCO 把它解释成"一个 GP 的均值"）：

$$
\pi(x) = \sum_{i=1}^{N}\beta_i\,\sigma_f\exp\!\Big(-\tfrac12 (x-c_i)^\top\Lambda(x-c_i)\Big),
\qquad \beta = (K+\sigma_n^2I)^{-1}T .
$$

形式上和 (3.3) 的 GP 均值**一模一样**，区别只是 $c_i$（中心点）、$T$（目标值）、
$\ell$（长度尺度）都是**可学习参数**。所以它的矩公式和 §4.3–4.6 完全平行：

$$
\phi_i = \sigma_f|I+\Lambda S|^{-1/2}\exp\!\Big(-\tfrac12\delta_i^\top\tilde\Lambda B^{-1}\tilde\Lambda\delta_i\Big),
\qquad \delta_i = m-c_i,
$$

$$
M = \beta^\top\phi,\qquad
\mathrm{Cov}[\pi] = \beta^\top\big(\Psi-\phi\phi^\top\big)\beta,\qquad
C = -\tilde\Lambda B^{-1}\tilde\Lambda\sum_i\beta_i\phi_i\delta_i .
$$

**唯一的区别**：策略的 $\sigma_f$ 固定为 1，噪声 $\sigma_n=0.01$ 只是数值正则化。

策略之后再过一个 `SaturationOp` 得到真正的 $u$。**这比原版 `conCat.py` 手推链式法则
简洁得多**——原版要把策略的矩与饱和函数的 Jacobian 相乘，这里只是两次 `splice`。

---

## 6. 规划：把矩向前滚

### 6.1 `plan()`

```python
m, S = initial.mean, initial.cov
for t in range(H):
    next_m, next_S, aug_m, aug_S = model.step_augmented(m, S)   # 高斯 → 高斯
    c, v = model.cost.moments(next_m, next_S)                   # 代价的矩
    m, S = next_m, next_S
```

$H$ 步之后得到 $\{m_t,S_t\}_{t=0}^{H}$、每步代价的均值 $c_t$ 与方差 $v_t$、
以及（可选的）均值 Jacobian $J_t$。

### 6.2 一步之内发生了什么

用 §5 的算子，`step_augmented` 就是**四次 `splice` + 一次投影**：

$$
\begin{aligned}
&\text{(1) } &z&\leftarrow[\,x;\ \sin x_a;\ \cos x_a\,] &&\text{TrigAugmentOp}\\
&\text{(2) } &z&\leftarrow[\,z;\ \pi(x)\,] &&\text{RbfPolicyOp}\\
&\text{(3) } &z&\leftarrow[\,z;\ \mathrm{sat}(u_{\text{raw}})\,] &&\text{SaturationOp}\\
&\text{(4) } &z&\leftarrow[\,z;\ f(x,u)\,] &&\text{GpDynamicsOp}\\
&\text{(5) } &x'&=Pz &&\text{投影}
\end{aligned}
$$

每一步都是"输入一个高斯、输出一个高斯"，而 `splice` 保证新块与旧块之间的相关性
被正确地写进去。**这就是为什么 `plan` 只有十几行**。

### 6.3 均值 Jacobian $J_t$

$$
J_t := \frac{\partial\, m_{t+1}}{\partial\, m_t}\Big|_{S_t\ \text{固定}} .
$$

它**不需要额外的求导**：由 §5.2 的仿射条件均值假设，

$$
\frac{\partial\,\mathbb{E}[g\mid x_{\text{in}}]}{\partial x_{\text{in}}} = C^\top
\quad\Longrightarrow\quad
\frac{\partial m_{t+1}}{\partial z} = P\,\big(\text{各步 }C^\top\text{ 的乘积}\big).
$$

代码在 `step_augmented(track_jacobian=True)` 里累乘 `res.cross.T @ Dm`。这是
"免费的 Jacobian"——因为 `cross` 本来就是条件均值的斜率。`tests/test_jacobian.py`
把它与 `torch.func.jvp` 对照过。

需要它的是 §7.2 的跨时刻协方差。不用它的目标（期望代价、信息增益）就传
`with_jacobians=False`，省下 5 倍多的时间（实测：`plan` 578 ms → 3160 ms）。

### 6.4 代价的累积

PILCO 的目标是

$$
J = \sum_{t}\gamma^{t+1}c_t .
$$

代码里 `Trajectory.value()` 就是这一行。注意**规划时**的 $c_t$ 是"转移之后的状态"
的代价，而**真实 rollout** 时 `rollout_true` 收的是转移之前的状态的代价——两者差一步。
这在对比"预测代价"与"实际代价"时会造成一点系统偏差，读数字时要知道。

---

## 7. 有向探索的数学

### 7.1 累积代价是一个随机变量

普通 PILCO 只优化 $\mathbb{E}[\sum_t c_t]$。但整条轨迹是随机的，所以
$C=\sum_t\gamma^{t+1}c(x_t)$ **本身是一个随机变量**。它有两个矩：

$$
\mu_C = \mathbb{E}[C], \qquad \sigma_C^2 = \mathrm{Var}[C].
$$

知道 $\sigma_C$ 之后，就可以像 UCB 那样做"乐观/悲观"的决策（§7.3）。

### 7.2 $\mathrm{Cov}(X_a,X_b)$ 的链式递推

$$
\mathrm{Var}\Big[\sum_t \gamma^{t+1}c_t\Big]
= \sum_t \gamma^{2(t+1)}\mathrm{Var}[c_t]
+ 2\sum_{t<s}\gamma^{t+s+2}\,\mathrm{Cov}[c_t,c_s].
$$

$\mathrm{Cov}[c_t,c_s]$ 需要两个时刻**状态**的联合分布，即 $\mathrm{Cov}[X_t,X_s]$。

**线性化**：把每一步看成 $X_{t+1}=A_tX_t+\text{const}$，其中 $A_t=J_t$（§6.3）。则

$$
\mathrm{Cov}(X_t,X_{t+1}) = S_tJ_t^\top,
$$

递推得

$$
\boxed{\;\mathrm{Cov}(X_t,X_{t+\tau}) = S_t\,J_t^\top J_{t+1}^\top\cdots J_{t+\tau-1}^\top\;}
\tag{7.1}
$$

**为什么是转置、为什么顺序是这样**：因为 $X_{t+1}$ 是"$X_t$ 经过 $J_t$"。协方差
在**左因子**上乘 $J^\top$。写成链式就是上面那样。

> ⚠️ 原版 `base/analytical_variance.py::compute_state_cross_covariances` 写成了
> $S_tJ_tJ_{t+1}\cdots$（没有转置、顺序也不对），实测相对误差 0.89–2.4。这是
> [`../pilco_python/docs/BUGS.md`](../pilco_python/docs/BUGS.md) 的 B1。

代码（`pilco/objectives.py`）：

```python
for tau in range(1, H):
    for t in range(H - tau):
        cov[t][t+tau] = cov[t][t+tau-1] @ J[t+tau].T     # (7.1)
```

### 7.3 三种探索目标

有了 $\mu_C,\sigma_C$：

| 名称 | 公式 | 含义 |
| --- | --- | --- |
| `mean` | $\mu_C$ | 普通 PILCO |
| `ucb` | $\mu_C + \beta\,\sigma_C$ | 乐观：偏好"不确定"的轨迹 |
| `gittins` | $\mu_C + \lambda\sigma_C$，$\lambda<0$ | **悲观**：偏好"确定"的轨迹 |

> ⚠️ **符号陷阱**：PILCO 的探索项是 $\mu_C + \lambda'\sigma_C$ 且 $\lambda'<0$，
> 所以它实际上是**下置信界（LCB）**，不是 UCB。文献里常写成 UCB 是因为
> "不确定 → 值得探索"的直觉，但在**最小化代价**的语境下，加正的 $\sigma$ 会
> 惩罚不确定性，方向反了。这一点 `docs/DIRECTED_EXPLORATION.md` 有详细考证
> （McAllister 2017 §4.3 vs Deisenroth  thesis §3.7.1）。

### 7.4 信息增益

另一个探索目标是"这条轨迹能让我学到多少"。GP 的互信息有闭式解：

$$
I\big(y;f\big) = \frac12\sum_{e=1}^{E}\log\det\Big(I+\frac{1}{\sigma_{n,e}^2}\bar K_e\Big),
\tag{7.2}
$$

其中 $\bar K_e[t,s]=\mathbb{E}[k_e(z_t,z_s)]$ 是**在计划的查询点上的期望核矩阵**。
它可以用 (4.1) 的同一个公式算：把 $z_t-z_s\sim\mathcal{N}(\delta,V)$ 当作一个
"均值 $\delta$、协方差 $V=\mathrm{Var}(z_t-z_s)$" 的随机变量，套 (4.1)。

**这个量在测什么**：(7.2) 是"看到 $n$ 个观测后，对函数 $f$ 的不确定性的减少量"。
如果所有查询点都挤在一起（核矩阵近似秩 1），$\log\det$ 小；如果它们互相独立
（核矩阵近似 $\sigma_f^2I$），$\log\det$ 大。

**为什么归一化**：$\bar K_e$ 的对角元恒为 $\sigma_f^2$（与策略无关），所以
(7.2) 里有一大块常数 $n\log(1+\sigma_f^2/\sigma_n^2)$。代码减掉这个常数再除以 $n$，
得到 $[0,\log(1+\sigma_f^2/\sigma_n^2))$ 内的可比值——"访问的点有多不冗余"。

**局限**：它是**无向**的。它奖励"覆盖不同的输入"，而不是"覆盖与任务相关的输入"。
所以单独用它会跑到无关的地方去；必须和代价项组合。

---

## 8. 导数：autograd 取代了什么

### 8.1 PILCO 手推了什么

原版 `gp0d.py`（233 行）里，**前三节是数学，后两节是微积分**：

| 行号 | 内容 | 能否用 autograd |
| --- | --- | --- |
| 80–107 | 缓存 $A^{-1},\beta$ | 可换成图内求解 |
| 116–219 | $\phi_n,\Psi_{ef},M,S,V$ 的闭式矩 | ❌ 不能——这是**积分**，不是导数 |
| 139–218 | `dMdm, dSdm, dVdm, dMds, dSds, dVds` | ✅ 完全能 |
| 226–233 | `order='F'` 向量化 | ✅ 不需要 |

`pilco_pytorch` 的 `GpDynamicsOp.moments` 只写前向，六个 Jacobian 全部由 autograd
给出。验证脚本 `scripts/diagnostics/autograd_vs_analytic_derivatives.py` 的结果：

```
dMdm  max|autograd - hand| = 1.7e-11      dSdm = 2.5e-11     dVdm = 7.4e-11
dMds  max|autograd - hand| = 6.3e-11      dSds = 3.0e-11     dVds = 见 §8.5
```

**六个里有五个在 $10^{-11}$ 量级一致**（这个量级正是"两套公式、不同运算顺序"的
舍入差）。

### 8.2 为什么 autograd 能做这件事（而为什么它不能做前一节）

**autograd 只会做一件事：链式法则。** 它把前向的每个 primitive 的偏导记下来，
然后反向（或前向）乘起来。

- 手推的六个 Jacobian **就是**链式法则的结果 → autograd 当然能重做。
- 但 $\phi_n=\mathbb{E}[k(x,X_n)]$ 这个**积分**不是任何东西的导数。如果你把
  "蒙特卡洛估计 $\frac1M\sum_m k(x^{(m)},X_n)$" 写成代码让 autograd 求导，
  你得到的是**那个估计量**的导数（高方差），不是精确矩的导数。**autograd 求导，
  它不积分。** 这是本文最想让你记住的一句话。

### 8.3 两个"白送"的恒等式

手写代码里有两条技巧，它们不是近似，而是**恒等式**：

$$
\boxed{\frac{\partial M}{\partial m} = V^\top}
\qquad\text{和}\qquad
\boxed{\frac{\partial V}{\partial m} = 2\Big(\frac{\partial M}{\partial S}\Big)^\top}
\tag{8.1}
$$

**第一条**：$V\equiv S^{-1}\mathrm{Cov}[x,f]$ 的定义恰好就是 $\partial M/\partial m$。
**第二条**：对高斯，$\partial/\partial S_{ij} = \frac12\partial^2/\partial m_i\partial m_j$，
所以 $\partial V/\partial m$ 与 $\partial M/\partial S$ 只差一个转置和因子 2。

所以六个 Jacobian 其实只有**四个独立**的。PILCO 手写代码利用这一点省了计算。
autograd 不需要知道这两条，但会算出同样的数（验证脚本里两条恒等式的残差是
$0$ 和 $2.8\times10^{-17}$）。

### 8.4 反向模式 vs 前向模式

| | 前向模式（`jacfwd` / JVP） | 反向模式（`backward` / VJP） |
| --- | --- | --- |
| 算的是 | $J\,v$（Jacobian 乘向量） | $J^\top \bar y$（Jacobian 转置乘向量） |
| 成本 | 输入的每一维一次 | 输出的每一维一次 |
| 适合 | 输入少、输出多 | 输入多、输出少 |

PILCO 的规划：输入是策略参数（几十个），输出是一个标量代价 $J$。**输出少、输入多
→ 反向模式**。所以 `optimise_policy` 里只是 `score.backward()`，一次就拿到
$J$ 对全部参数的梯度。

但 §6.3 的 $J_t$ 是"4 个输出对 4 个输入"的**完整 Jacobian**，反向模式要 4 次，
前向模式也要 4 次，差不多；代码用 `torch.func.jacfwd`（实测是普通 `plan` 的 5.46 倍）。

### 8.5 一个真实的坑：对称矩阵上的导数约定

§1.4 提过：手写的 $\partial V/\partial S$ 是**对称化**过的，autograd 的**不是**。
实测差 $6.3\times10^{-3}$，但

- 对称化之后两者差 $7\times10^{-11}$；
- 沿**对称方向**做有限差分，两者都与真值差 $8\times10^{-8}$。

**结论**：两者都是正确的方向导数，只是"把导数摊在 $(j,k)$ 还是 $(k,j)$"不同。
因为下游的 $\Delta S$ 永远是对称的，这个差别在缩并时消失。

**对读代码的启示**：如果你看到 autograd 的 $\partial S/\partial s$ 与手写代码对不上，
先检查是不是这个约定问题，再怀疑 bug。

### 8.6 图在哪里被切断

`optimise_policy` 里有

```python
for p in gp_params:
    p.requires_grad_(False)
```

**为什么**：策略优化时把 GP 超参数当常数。这不是偷懒，而是**正确的算法结构**：
PILCO 是"先拟合模型，再在模型里优化策略"，模型在这一阶段不该变。如果不断开，
梯度会试图通过 GP 超参数回传，那是在优化一个错误的联合目标。

同理，GP 训练时策略参数也要断开。**"在哪切断"是设计决策，不是实现细节。**

### 8.7 成本账

用 `scripts/diagnostics/autograd_vs_analytic_derivatives.py` 实测（同一个 GP 问题，
$n=30,D=3,E=2$，以"前向一次"为单位）：

| 方式 | 相对成本（约） |
| --- | --- |
| 手写完整 Jacobian | $2.1\times$ 前向 |
| `jacfwd` 完整 Jacobian | $4.5\text{–}6.4\times$ 前向 |
| `backward` 一次 VJP | $1.3\text{–}3.0\times$ 前向 |

（区间来自重复运行：torch 在 $3\times3$ 矩阵上的固定开销让单次计时抖动较大，
但排序稳定。）

**结论**：

- 当目标是**标量**时（策略优化），autograd 的 VJP 与手写代码**同一量级**。
- 当需要**完整 Jacobian** 时（§6.3），autograd 贵约 2 倍，因为它老老实实算了
  $\partial S/\partial s$ 这种标量目标用不到的块。
- 绝对耗时不可比：torch 在 $3\times3$ 矩阵上每次算子调用有固定开销，前向就比
  NumPy 慢 10 倍以上。**这是实现常数，不是算法差异。**

所以重写的收益**不是速度，而是正确性与可维护性**：

1. 不用为每个新算子手推 6 个 Jacobian（原版每个 `gp*d.py` 有一半是导数代码）；
2. 不用维护 `order='F'` 的索引体操（原版最容易出错的地方）；
3. 二阶导（Hessian）自动可得，而手写代码要再写一遍。

---

## 9. 公式 ↔ 代码对照表

| 公式 | 代码位置 | 备注 |
| --- | --- | --- |
| (1.1) 二次型期望 | `cost.py::_e1` | `logdet(A)` + 解线性方程 |
| (1.2) 带线性项 | `gp.py::moments` 的 `maha` 项 | $b=-\Lambda\delta$ |
| (3.2) GP 预测 | `gp.py::_solve_cache` | $\beta=A^{-1}Y$ 缓存 |
| (3.4) 边缘似然 | `gp.py::nlml` | 三项：拟合/复杂度/常数 |
| (4.1) $\phi_n$ | `gp.py::moments` 的 `ell` | 注意变量名叫 `ell` |
| (4.2) $\Psi_{ef}$ | `gp.py::moments` 的 `L_mat` | `maha(ii,-ij,solve(R,s)/2)` |
| (4.3) 输出协方差 | `gp.py::moments` 的 `S` 组装 | 三项：先验/信息/输入不确定 |
| (4.4) 交叉协方差 | `gp.py::moments` 的 `cross` | `-iell * solve(B, iell*g)` |
| (5.1)(5.2) 三角矩 | `fourier.py::_harmonics_moments` | 频率靠 `Harmonic.freq` |
| (5.3) 三角交叉 | `fourier.py` 的 `out_cross` | 稀疏：只在角坐标上 |
| (5.4) 饱和 | `fourier.py::SaturationOp` | 两个 Harmonic + readout |
| (5.5) 代价矩 | `cost.py::moments` | $e_1,e_2$ |
| §5.2 拼接 | `base.py::splice` | `proj = cov[:,in] @ cross` |
| §6.2 一步 | `trajectory.py::step_augmented` | 四次 splice + 投影 |
| (7.1) 跨时刻协方差 | `objectives.py::cost_cross_covariances` | `@ J[t+tau].T` |
| (7.2) 信息增益 | `objectives.py::information_gain` | `logdet(I + Kbar/σ²)` |
| (8.1) 两个恒等式 | — | autograd 自动满足 |

### 9.1 三个最容易读错的地方

1. **`ell` 不是长度尺度**。`ell` / `ell_cols` 是 $\phi_n=\mathbb{E}[k(x,X_n)]$；
   长度尺度是 `lengthscales` / `exp(log_lengthscales)`。
2. **`maha(a,b,M)` 是在算指数里的二次型**，$M$ 通常是 $R^{-1}S/2$ 这种"半个协方差"。
3. **`R` 不是协方差**。在 GP 里 $R=I+S(\Lambda_e+\Lambda_f)$ 是"两个核相乘后"的
   归一化矩阵；它的行列式就是 $|I+AS|$。

---

## 10. 一页速查

$$
\begin{aligned}
&\text{高斯条件化：} && m_b + S_{ba}S_{aa}^{-1}(v-m_a),\quad S_{bb}-S_{ba}S_{aa}^{-1}S_{ab}\\
&\text{仿射推送：} && Fm+b,\quad FSF^\top\\
&\text{二次型期望：} && \mathbb{E}[e^{-\frac12x^\top Wx}]=|I+SW|^{-1/2}e^{-\frac12m^\top W(I+SW)^{-1}m}\\
&\text{核期望：} && \phi_n=\sigma_f^2|B|^{-1/2}e^{-\frac12\delta_n^\top\tilde\Lambda B^{-1}\tilde\Lambda\delta_n},\quad B=I+\tilde\Lambda S\tilde\Lambda\\
&\text{两核期望：} && \Psi_{ef}[n,n']=\sigma_f^2\sigma_f^2|I+AS|^{-1/2}e^{k_n+k_{n'}+\frac12d^\top G d}\\
&\text{输出协方差：} && \delta_{ef}\sigma_f^2-\delta_{ef}\mathrm{tr}(A^{-1}\Psi_{ee})+\beta_e^\top(\Psi_{ef}-\phi_e\phi_f^\top)\beta_f\\
&\text{拼接：} && \mathrm{Cov}[z,g]=\Sigma[:,in]\,C,\quad C=S^{-1}\mathrm{Cov}[x,g]\\
&\text{三角：} && \mathbb{E}[\sin a]=e^{-\sigma^2/2}\sin\mu,\quad \mathrm{Cov}[x_j,\sin a]=S_{ja}e^{-\sigma^2/2}\cos\mu\\
&\text{饱和：} && \mathrm{sat}(u)=u_{\max}(9\sin u+\sin 3u)/8\\
&\text{代价：} && \mathbb{E}[c]=1-e_1,\quad \mathrm{Var}[c]=e_2-e_1^2\\
&\text{跨时刻：} && \mathrm{Cov}(X_t,X_{t+\tau})=S_tJ_t^\top\cdots J_{t+\tau-1}^\top\\
&\text{信息增益：} && \tfrac12\sum_e\log\det(I+\sigma_{n,e}^{-2}\bar K_e)
\end{aligned}
$$

**如果只记三句话**：

1. 世界是高斯，高斯在仿射和条件化下封闭——所以每一步都能"闭式"地推。
2. GP 预测就是在不确定输入下算 $\mathbb{E}[k]$ 和 $\mathbb{E}[kk^\top]$，靠的是
   "高斯乘高斯还是高斯"。
3. autograd 只替代**导数**，不替代**积分**；手推的 6 个 Jacobian 可以全部删掉，
   但 $\phi_n,\Psi_{ef}$ 这些闭式矩必须自己写。
