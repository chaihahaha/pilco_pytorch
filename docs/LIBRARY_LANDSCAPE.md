# 能简化 PILCO 的现成库：候选清单与实测证据

> 这份文档回答一个问题：`pilco_pytorch` 里哪些代码本来可以不必自己写，用现成库代替？
> 每个候选都标注**能删掉本仓库的哪些文件/函数**，以及**哪些部分谁也替不了你**。

## 0. 先说在你自己的项目里找到的证据

对 `~/source` 下所有 `.py` 做过导入扫描（`sensor_kalman_rl`、`sensor_opt_rl`、
`sensor_pilco`、`world_model_fisher_eigen`，并排除 vendored 大仓库），第三方 GP 库
**只出现过一处**：

```
world_model_fisher_eigen/active_pilco.py:3:   import gpytorch
world_model_fisher_eigen/active_pilco.py:8:   class DynamicsGP(gpytorch.models.ExactGP)
```

另外 `sensor_pilco/MC-PILCO/environment.yaml` 里列了 `gpytorch`（但 MC-PILCO 的
`gpr_lib/` 实际是**自己手写的**纯 PyTorch GP，没有真的用它）。

所以"我尝试过的 GP 库"最可能就是 **GPyTorch**。`pip` / `conda` / `uv` 缓存和三个
conda 环境里都没有 `gpflow`/`gpytorch`/`emukit`/`probnum`/`botorch` 的残留，
说明它们没有在这台机器上装成功过（或早已清理）。

---

## 1. 候选清单

### 1.1 GPyTorch —— 替掉"GP 模型层"

你已经用过。它提供 `ExactGP`/`SingleTaskGP`、`RBFKernel(ard_num_dims=)`、
`ScaleKernel`、`GaussianLikelihood`、`ExactMarginalLogLikelihood`，以及基于
`LinearOperator` 的结构化线性代数。

| 本仓库被替掉的 | 位置 |
| --- | --- |
| GP 的 NLML 与超参拟合 | `pilco/models/gp.py::nlml` / `fit_hyperparameters` |
| 手写 Cholesky 缓存与抖动 | `pilco/ops/gp.py::_solve_cache`、`gaussian.py::safe_cholesky` |
| 核函数实现 | `pilco/ops/gp.py::_log_ell`、`share_lengthscales` |
| L-BFGS/curb 那段调参逻辑 | `models/gp.py::curbed_nlml`（curb 需要自己保留） |

**替不掉**：不确定输入下的矩（`ops/gp.py::moments`）。GPyTorch 只在确定点上求值。

### 1.2 TorchRL 的 `GPWorldModel` —— **最接近"PILCO 现成实现"的库**

`torchrl.modules.models.gp.GPWorldModel`（torchrl ≥ 0.12，实测 0.12 文档里已有，
当前 PyPI 最新 0.14.1）。它的 docstring 直接写着"Implements the probabilistic
dynamics model from PILCO (Deisenroth & Rasmussen, 2011)"，并且：

- `fit()`：每个状态维一个 `SingleTaskGP`（ARD RBF + `GammaPrior`），用 BoTorch 的
  `fit_gpytorch_mll` 做证据最大化；
- `forward()`：输入带协方差时走 `uncertain_forward()`，**按论文 Eq. 10–23 做精确矩匹配**：
  `q_a`（核均值）、`Q_ab`（跨核矩阵）、`α²_a − tr(K⁻¹Q_aa)` 的对角修正、
  `cov[x̃,Δ]`、以及 `Σ_t = Σ_x + Σ_Δ + cov + covᵀ` 的组装；
- 输入是 `(μ_x, Σ_x, μ_u, Σ_u, C_xu)`，输出 `(μ_t, Σ_t)`。

| 本仓库被替掉的 | 位置 |
| --- | --- |
| 整个 GP 矩匹配算子 | `pilco/ops/gp.py::moments`（约 120 行，最难的部分） |
| 一步的协方差传播 | `pilco/trajectory.py::step_augmented` 的第 4、5 步 |
| GP 拟合与超参 | `pilco/models/gp.py` 全部 |
| `mean_jacobian`（它用 `C_xu` 交叉协方差解析地给出） | `trajectory.py::mean_jacobian` |

**替不掉**：三角增广、饱和、代价的矩、`splice` 的增广布局（它是 state 和 action
分开输入的，没有 `[x | sin,cos | u_raw | u | dyn]` 这一套）。

### 1.3 GPflow 的 `uncertain_conditional` —— 同一件事的 TensorFlow 版本

`gpflow/conditionals/uncertain_conditionals.py`。它的核心是

```python
eKuf  = expectation(pXnew, (kernel, inducing_variable))          # ψ1 = E[k(x, X_n)]
eKff  = expectation(pXnew, kernel)                               # ψ0 = E[k(x,x)]
eKuffu= expectation(pXnew, (kernel, iv), (kernel, iv))           # ψ2 = E[k(x,X_n)k(x,X_m)]
...
fvar  = (eKff - trace(Li_eKuffu_Lit))[:, None] + ... - fmean**2
```

`ψ1`、`ψ2` 就是本仓库的 `φ_n`、`Ψ_ef`，`fmean`/`fvar` 就是 `Moments.mean`/`cov`。
换句话说 **GPflow 已经有 `gp0d.py` 的数学**，包括 `full_output_cov=True` 给出的
$E\times E$ 全协方差。

| 被替掉的 | 位置 |
| --- | --- |
| `ops/gp.py` 的全部矩代码 | 同上 |
| 核的解析期望（RBF/Matern × 高斯） | `expectation` 模块 |

**代价**：它是 TF2 + 变分（`q_mu`/`q_sqrt`）接口，移植到 PyTorch 项目要跨框架。

### 1.4 Emukit 的 Bayesian quadrature —— "核嵌入"那一半

`emukit.quadrature.kernels.QuadratureRBFGaussianMeasure`：

- `qK(x2)` = **核均值** $\int k(x,x_2)\,p(x)\,dx$ —— 正是 `φ_n`；
- `qKq()` = 双边积分；
- `GaussianMeasure` / `LebesgueMeasure` 指定积分测度。

**替不掉**：`Ψ_ef`（两个**不同**训练点的核乘积的期望）—— Emukit 的 `qKq` 是同一个
测度积两次，不是跨点版本，需要自己按 §4.4 组装。

### 1.5 ProbNum —— 滤波视角的"矩匹配"

`probnum.quad`（Bayesian quadrature）＋`probnum.filtsmooth`（贝叶斯滤波/平滑）。
PILCO 的 `plan()` 本质就是"反复做假定密度滤波（assumed density filtering）"，
ProbNum 把这件事抽象成 `filtsmooth` 的 transition + 高斯矩更新。
如果想换个框架重写规划器，这是最接近的理论对应物；但它不提供 GP 核嵌入的闭式
`Ψ`，仍要自己写。

### 1.6 MC-PILCO（你仓库里就有）

`sensor_pilco/MC-PILCO/`（上游 `merlresearch/MC-PILCO`）。它**用粒子采样替代解析矩**：
把 $x_t$ 表示成粒子集，推过 GP 后取经验均值和协方差，用 dropout 控制方差增长。

| 它能删掉 | 代价 |
| --- | --- |
| `ops/gp.py::moments`、`mean_jacobian`、`objectives.py` 的累积方差 | 需要几百个粒子 + 调 dropout；单步方差有采样噪声 |
| `optim/minimize.py`（改用 Adam + 自动微分） | 精确性从 $10^{-12}$ 掉到 $O(1/\sqrt{M})$ |

### 1.7 `prob_mbrl`（McGill）

`mcgillmrl/prob_mbrl` —— "probabilistic model based RL in PyTorch"，README 明确写
是 Deep-PILCO / MC-PILCO 的复现。适合当参考实现读，但成熟度一般（README 说
"in progress"，还依赖 `kusanagi`）。

---

## 2. 一个都替不了你的部分

这是重写时最容易误判的地方。以下东西**没有任何库提供**，因为它们是 PILCO 特有的：

1. **三角增广**（`ops/fourier.py::TrigAugmentOp`）：$E[\sin a]$、$\mathrm{Cov}[\sin a,\cos b]$。
   通用 GP 库不管这个。
2. **饱和函数**（`SaturationOp`）：$u_{\max}(9\sin u+\sin3u)/8$ 的矩。
3. **代价的矩**（`ops/cost.py`）：$E[1-e^{-\frac12 d^\top Wd}]$ 与跨时刻的
   $E[c(x_1)c(x_2)]$。
4. **`splice` 与增广布局**：把五个槽位拼成一个联合高斯并保证相关性正确。
5. **有向探索目标**：累积代价方差、闭式信息增益、UCB/Gittins。

换句话说：**GP 那一层（含不确定输入矩）可以外包，PILCO 的"任务层"不行。**

---

## 3. 如果只做一件事

按"改动量 ÷ 收益"排序：

1. **换 GP 模型层为 GPyTorch**（低风险）：删掉 `models/gp.py` 的 NLML 与
   `ops/gp.py::_solve_cache`，保留 `moments` 的数学，只把 `K`/`β` 的构造交给
   `SingleTaskGP`。这一步能删约 200 行。
2. **把 `ops/gp.py::moments` 换成 torchrl `GPWorldModel.uncertain_forward` 的移植**
   （中风险）：能删掉最难的那 120 行，但要注意它输入是 state/action 分开的，
   而 PILCO 需要 $[x,\sin,\cos,u]$ 的增广输入 —— 移植时要补一层投影。
3. **不要**为了"简化"改用 MC-PILCO 的粒子路线再重跑 cartPole：本仓库的解析矩在
   `n=320` 时耗 13.7 ms/步且精确到 $10^{-12}$，粒子法要在这个精度上需要
   $O(10^6)$ 粒子。

---

## 4. 一句话总结

- 你**用过**的是 **GPyTorch**（`world_model_fisher_eigen/active_pilco.py` 里）。
- 你**可能记住的"能极大简化 PILCO"的东西**，大概率是下面二者之一：
  - **TorchRL `GPWorldModel`**（PyTorch 官方，按 PILCO 论文 Eq. 10–23 实现了完整矩匹配）；
  - **GPflow `uncertain_conditional`**（TF，`expectation` 直接给 ψ0/ψ1/ψ2）。
- 二者都只替掉 **GP 矩匹配**，都不管三角增广/饱和/代价/探索目标 —— 那部分是本仓库
  存在的理由。
