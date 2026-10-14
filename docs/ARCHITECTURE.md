# Architecture of `pilco_pytorch`

This document explains what the PyTorch rewrite changes relative to
`pilco_python` (the NumPy port of the MATLAB original), why the new structure is
shorter and less error-prone, and how to extend it.

---

## 1. The one idea

Everything PILCO does analytically is an instance of a single pattern.  Given a
Gaussian input `x ~ N(m, S)`, a nonlinearity `g`, and PILCO's *conditional
linearity* assumption, an operator must report three things:

| quantity | meaning | PILCO's name |
| --- | --- | --- |
| `E[g(x)]` | output mean | `M` |
| `Cov[g(x)]` | output covariance | `S` / `V` |
| `C = S⁻¹ Cov[x, g(x)]` | input–output regression matrix | `C` / `V`, `Q`, `R` |

The third quantity is the load-bearing one: it lets the output block be *spliced*
into a larger Gaussian while carrying the correlations with every other variable
along, which is what makes closed-form multi-step planning possible.

In `pilco_python` every operator (`gTrig`, `gSin`, `gSat`, `congp`, `gp1d`,
`lossSat`, `propagated`, …) implements this pattern **and** the derivatives
`dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp, dSdp, dCdp`, hand-derived and
hand-vectorised in MATLAB's Fortran-`vec` convention.  That derivative code is
roughly half of the code base and is where essentially all subtle bugs live.

In `pilco_pytorch` an operator only implements the **forward** moments as a
differentiable function of `(m, S, parameters)`.  Autograd produces every
derivative — including second-order ones, which the NumPy code never had.

```mermaid
flowchart LR
    A["moment ops<br/>(gTrig, GP, RBF, sat, cost)"] -->|"E[g(x)], Cov[g(x)], C"| B["splice()"]
    B --> C["augmented Gaussian"]
    C --> D["plan(): H steps"]
    D --> E["objectives:<br/>value / variance / info gain"]
    E -->|"autograd backward"| A
```

### What that buys, concretely

| NumPy `pilco_python` | `pilco_pytorch` |
| --- | --- |
| `base/propagated.py` 265 lines, incl. a duplicated derivative variant | `PilcoModel.step_augmented` ≈ 40 lines, one code path |
| `_fill_in` with 5 Fortran-index bookkeeping blocks | `ops/base.py::splice`, 10 lines, no index arithmetic |
| `util/gTrig.py` + `util/gSin.py` + `util/gSat.py` (376 lines total) | `ops/fourier.py` (≈ 250 lines incl. docs), one shared harmonic-moment helper |
| `gp/gp0d.py` (233 lines, incl. `dMdm…dVds`) | `ops/gp.py::moments` (≈ 70 lines, forward only) |
| `gp/train.py` + `gp/hypCurb.py` + `gp/covSEard.py` + `gp/covSum.py` + `gp/covNoise.py` + `gp/fitc.py` | `models/gp.py` (~120 lines) — L-BFGS on the NLML |
| `loss/lossSat.py` (116 lines incl. derivatives) | `ops/cost.py::SaturatingCostOp._e1/_e2` (12 lines) |
| `util/minimize.py` (418-line MATLAB-style BFGS) | `torch.optim.LBFGS` |
| the entire `dMdp`/`dSdp` policy-gradient chain (`congp` 120 lines, `conCat` 150, `conlin` 104, `gp2d` 411) | nothing — the policy is an `nn.Module` and `loss.backward()` works |

---

## 2. Modules

```
pilco/
  gaussian.py     Gaussian container + numerically safe linear algebra
  ops/
    base.py       MomentOp protocol, Moments container, splice()
    fourier.py    sin/cos harmonics: gTrig + gSin/gSat in one abstraction
    policy.py     RBF controller (congp + gp2)
    gp.py         ARD squared-exponential GP dynamics (gp0d)
    cost.py       saturating cost (lossSat) + trig-augmented variant (loss_cp)
    joint.py      joint trig augmentation of two correlated states
  trajectory.py   StateSpec, PilcoModel, plan(), mean Jacobians
  objectives.py   value, cumulative-cost variance, information gain, acquisitions
  models/gp.py    hyper-parameter training
  pilco.py        the learning loop
  scenarios/      cartpole.py, mountaincar.py
tests/            validation against the NumPy reference and Monte Carlo
scripts/          run_cartpole.py, benchmark_exploration.py
```

### `ops/base.py` — `MomentOp` and `splice`

```python
class MomentOp(nn.Module):
    n_in: int
    n_out: int
    def moments(self, mean, cov) -> Moments: ...      # E[g], Cov[g], C
```

`splice(mean, cov, op, in_idx, out_idx)` attaches an operator's output to an
augmented Gaussian:

```python
res   = op(mean[in_idx], cov[in_idx][:, in_idx])
proj  = cov[:, in_idx] @ res.cross          # Cov[x_j, g] = Cov[x_j, x_in] S_in⁻¹ Cov[x_in, g]
mean  = mean.index_copy(0, out_idx, res.mean)
cov[:, out_idx] = proj; cov[out_idx, :] = proj.T; cov[out_idx, out_idx] = res.cov
```

That is the whole of `base/propagated.py::_fill_in` plus its cross-covariance
bookkeeping, without the Fortran index gymnastics.

### `ops/fourier.py` — one abstraction for four operators

Two facts replace `gTrig`, `gSin`, `gSat` **and** their derivatives:

* For jointly Gaussian `(u, v)` with `Var(u) = a`, `Var(v) = b`, `Cov(u,v) = c`,
  `E[sin u]`, `E[sin u sin v]`, `E[sin u cos v]`, … all have closed forms.
* Stein's lemma gives the input–output covariance:
  `Cov[x_j, sin(f x_i)] = f S_{ji} e^{-f² S_{ii}/2} cos(f m_i)`.

Treating `sin(f x_i)` as a random variable with mean `f m_i` and variance
`f² S_{ii}` makes frequency-3 terms such as `sin(3x)` fall out of the same
formulas, so `gSat`'s cubic-ish saturation `maxU(9 sin u + sin 3u)/8` is just a
readout matrix over harmonics.  A degenerate `Harmonic` list can express any
finite Fourier nonlinearity, which is how new squashing functions are added.

### `trajectory.py` — the augmented state

For state dimension `D`, `A` angles and `Du` controls, one step builds a Gaussian
over the slot layout (identical to PILCO's `M`/`S`):

```
[ state (D) | sin/cos (2A) | u_raw (Du) | u (Du) | dyn (D) ]
   idx_state    idx_trig      idx_u_raw    idx_u    idx_dyn
```

and fills it with five splices — `gTrig`, controller, saturation, GP, projection.
`plan()` repeats this for `H` steps and returns the per-step Gaussians, the
per-step costs and variances, and the mean Jacobians `J_t = ∂m_{t+1}/∂m_t`.

---

## 3. Gradients: what autograd replaces

* **Policy gradients** (`dMdp/dSdp/dCdp` chains in `congp`/`conCat`/`conlin`):
  gone.  `loss.backward()` works because the controller is an `nn.Module` and
  the splices are differentiable.
* **Second-order gradients**: free.  `pilco_python` has none; they are needed for
  Newton-style policy optimisation and for the exact Hessian of the objective.
* **GP hyper-parameter gradients** (`gp/hypCurb.py`): replaced by
  `GpDynamicsOp.nlml()` plus `torch.optim.LBFGS`.
* **The mean Jacobian** `J_t` (needed for cross-timestep covariances):
  `torch.func.jacfwd` over the actual forward computation.

A trap worth recording, because it cost real debugging time: `J_t` is **not**
the chain of the operators' `Cᵀ` blocks.  `C` gives `∂E[g]/∂m`, but each
operator's mean also depends on the input covariance (`E[sin x] = e^{-v/2} sin m`)
and the intermediate covariances depend on the state mean.  PILCO carries this as
the separate `dMds`/`dSdm` blocks.  `mean_jacobian_jvp` (forward-mode over the
real computation) is kept as the oracle, and `tests/test_end_to_end_vs_numpy.py`
checks the two agree.

---

## 4. Numerical lessons

Four issues found while writing this, all of which the NumPy reference shares or
hides:

1. **Symmetrising a non-symmetric determinant.**  PILCO's Gaussian integrals
   constantly produce `I + S W` with symmetric `S` and `W` — but `S W` is *not*
   symmetric, and `det(I + S W)` is not `det(I + (SW + WS)/2)`.  `gaussian.logdet`
   uses `slogdet` for general matrices; `logdet_psd_scaled_diag` uses the
   identity `det(I + SΛ) = det(I + Λ^{1/2} S Λ^{1/2})` for diagonal `Λ` to keep a
   Cholesky-friendly symmetric form.
2. **The product-kernel quadratic form.**  In `E[k(x,X_n) k(x,X_m)]` the `n = m`
   and cross pieces must be combined as `d_n + d_m` (a diagonal plus a full
   outer product); writing `R G R` for all three terms silently multiplies the
   off-diagonal by 4.  This produced a 20% error in the GP predictive
   covariance until it was caught by Monte-Carlo comparison.
3. **The cost cross-moment.**  `E[c(x₁)c(x₂)] = 1 - E[e^{-a₁}] - E[e^{-a₂}] +
   E[e^{-a₁-a₂}]` — the *exponential* expectations, not the cost expectations.
   Using `E[c]` flips the sign of the entire cross term.  The first version of the
   unit test encoded the same mistake and "passed"; the test now compares against
   a Monte-Carlo sample of the joint Gaussian, which cannot share the bug.
4. **Pairwise cross-covariances are not jointly consistent.**  The linearised
   chain product `S_t J_t^T J_{t+1}^T …` gives correct pairwise cross-covariances
   but the assembled `H·D × H·D` joint can be slightly indefinite.  It must be
   projected to the PSD cone before sampling (the planner itself never forms the
   full joint, so this only matters for validation and for the log-det objective).
5. **Do not symmetrise a cross-covariance between two different vectors.**
   `Cov[sin u, cos v] ≠ Cov[cos u, sin v]` in general — they differ by
   `sin(μ−ν) e^{−(p+q−2r)/2}`.  `ops/joint.py::_trig_trig_cross` initially called
   `symmetrize` on its output, which corrupted the trig–trig cross block for any
   scenario whose cost mixes `sin` and `cos` (cart-pole does).  With a single
   angle the two entries differ, so even `A = 1` was affected; the error only
   became visible once `tests/test_joint.py` compared in Monte-Carlo standard
   error units instead of by absolute magnitude.

---

## 5. Validation

| test | what it proves |
| --- | --- |
| `tests/test_ops_vs_numpy.py` | `TrigAugmentOp` ≡ `gTrig`, `SaturationOp` ≡ `gSat`, `GpDynamicsOp` ≡ `gp0d`, `SaturatingCostOp` ≡ `lossSat` (machine precision); cost cross-moment ≡ Monte Carlo |
| `tests/test_end_to_end_vs_numpy.py` | `plan().value()` ≡ `base/value.py` on a real cart-pole GP (3×10⁻⁸ relative); autograd gradient ≡ PILCO's hand-derived `dJ/dp` (6×10⁻⁶); `jacfwd` Jacobian ≡ `jvp`; cumulative-cost variance ≡ joint-Gaussian Monte Carlo |
| `scripts/run_cartpole.py` | the loop actually learns cart-pole |

The NumPy side is the ground truth: `pilco_python/tmp/gradcheck_value.py` first
verifies that *its* analytic gradient matches finite differences (relative error
1.8×10⁻⁶), so the cross-implementation comparison is meaningful.

---

## 6. Extending: adding a new operator

```python
class MySquash(MomentOp):
    def __init__(self, n_in, channels):
        super().__init__()
        self.n_in, self.n_out = n_in, len(channels)
        self.harmonics = [Harmonic(c, 1.0, SIN, 1.0) for c in channels]

    def moments(self, mean, cov):
        return _harmonics_moments(mean, cov, self.harmonics)   # closed form
```

Then splice it into `PilcoModel.step_augmented`.  There is nothing else to
write: no Jacobians, no `vec` conventions.  If the nonlinearity is not a Fourier
sum, implement `moments` directly with whatever Gaussian integrals are available
and let autograd handle the rest; only the forward pass must be closed form and
differentiable.

## 7. Extending: adding a scenario

A scenario needs `spec: StateSpec`, `cost`, `initial_state()`,
`optimisation_state()`, `step_true()`, `rollout_true()` and
`data_from_rollout()`.  See `scenarios/mountaincar.py` for a minimal example
(the loop in `pilco.py` is entirely scenario-agnostic).
