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

---

## 8. Known gap: the closed loop does not yet match the reference

The model layer is validated; the learning loop is not at parity on cart-pole,
and it is worth recording precisely what is and is not known.

**What matches.**  All operators, the planner, the value and its gradient agree
with the NumPy reference (see the validation table in `README.md`).  On
*identical* data and an identical starting policy the policy optimiser is
comparable: NumPy `minimize` reaches `J = 32.37` in 151 line searches, torch
L-BFGS reaches `J = 33.34` in 187 evaluations
(`scripts/diagnostics/optimiser_parity.py`).

**What does not.**  The closed loop reaches 19–38 per `H = 40` cart-pole
episode depending on settings, against the reference's ≈ 9 after one learning
episode, with high seed variance.

**Diagnosis, stage 1 — the dynamics fit.**  `scripts/diagnostics/gp_fit_effect.py`
holds the policy optimiser, the starting controller and the 40-point dataset
fixed and varies only the GP fit:

| variant | GP NLML | policy J after the same optimiser |
| --- | --- | --- |
| A. hyper-parameters from NumPy `minimize` | `+51.3` | **33.85** |
| B. torch L-BFGS, 150 iterations | `−17.6` | 35.80 |
| C. torch L-BFGS, 30 iterations | `−17.9` | 35.95 |
| D. torch L-BFGS, 150 iterations, ARD lengthscales shared across outputs | `+3.7` | 38.29 |

So the GP fit *is* a real effect (≈ 2 points, 5.8%), but it does **not** by itself
explain the closed-loop gap (reference ≈ 9 vs ours 19–38).  Note also that
reducing the hyper-parameter count (D) makes things worse, so "fewer parameters"
is not the fix, and B ≈ C shows that early stopping is not either: the torch
optimum is simply a different, less useful fit than NumPy's.

**Diagnosis, stage 2 — everything else checks out.**  The scenario plumbing was
verified against the reference: `CartPole.policy_numpy` reproduces
`conCat(congp, gSat)` to 5.4e-6, and the planner value/gradient agree as in the
validation table.  (An apparent trajectory mismatch turned out to be my own test
using `np.random.default_rng` where the reference uses `np.random.seed` — a
different noise stream, not a bug.)

**Stage 2b — the operators are not at fault, and neither is autograd.**
``scripts/diagnostics/optimizer_trace.py`` re-evaluates the PyTorch model at
*every one* of the 57 objective evaluations PILCO's BFGS made on the same data::

    max |J_torch - J_numpy|     = 1.299e-05
    max rel |g_torch - g_numpy| = 1.542e-04

and, started from NumPy's final parameters, torch's L-BFGS reaches a *lower*
objective (16.98 vs 17.24).  So there is no autograd drift and the policy
optimiser is not weaker.  ``scripts/diagnostics/gp_training_parity.py`` checks
the GP objective the same way and finds it **identical per output** (numpy
``gpr`` and torch ``nlml`` agree to 1e-9 at the same hyper-parameters; an earlier
apparent 159-unit discrepancy was my own comparison of one output against the
sum of four).  The genuine difference is that torch's L-BFGS converges further
than PILCO's MATLAB-style ``minimize`` and lands at NLML -17.6 where the
reference stops at +51.3.

**Stage 3 — the loop does learn; it is not stable.**  Running the loop for
eight episodes with `gp_iterations=30, policy_iterations=60` (recorded in
`results/cartpole_learning_gp30.json`):

| episode | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| realised cost | 35.6 | 30.3 | **11.7** | **12.2** | 40.0 | 38.1 | 39.9 | 40.0 |
| planned value | 35.2 | 32.3 | **14.1** | 29.1 | 37.9 | 37.9 | 37.8 | 37.9 |

Episodes 3–4 reach 11.7 / 12.2 against the NumPy reference's floor of 8.3–12.8,
so **cart-pole is solved**.  It then collapses to the cost ceiling (40.0) and
stays there.  Note the planned value in episode 4 (29.1) already warns of the
collapse: the model has stopped predicting the good policy's cost correctly.

**Stage 4 — episode-by-episode reproduction.**
``scripts/diagnostics/reproduction_check.py`` diffs the two implementations at
each level of the loop, with the RNG routed through the same stream::

    1) initial dataset / value
       datasets identical        : True
       J numpy=39.3781746773  torch=39.3781753472  diff=6.70e-07
    2) GP fit on that dataset      <-- the divergence
       numpy  log ell[out0] = [5.1506 2.0613 5.9450 2.9242 3.6780]
       torch  log ell[out0] = [2.7337 1.1466 2.5401 0.0022 2.7201]
       hyper-parameter max abs diff = 3.4049
       -NLML  numpy -51.2716   torch -17.6107
    3) policy value after the same optimiser
       reference fit 33.8543   torch fit 35.8010

So the *only* level at which the two implementations disagree is the point the
GP fit stops at.  The reference's fit is much **smoother** (log-lengthscales up
to 5.95 against the input spread's 1.73) than a converged fit (2.73 against the
same 1.73), and at 40 data points the smoother model yields the better policy.

**Closing the gap (measured).**  `scripts/diagnostics/followup_experiments.py
--part A` sweeps the lengthscale curb (``ls_range``, in log units from
``log std(inputs)``):

| fit | policy value |
| --- | --- |
| reference hyper-parameters (no fitting) | **33.85** |
| `ls_range=None` (PILCO's own curb) | 35.80 |
| `ls_range=3.0` | **34.91** |
| `ls_range=2.0` | 35.71 |
| `ls_range=1.5` | 36.01 |

`ls_range=3.0` recovers about half of the gap; the response is not monotonic, so
it is exposed as an option (`PilcoConfig.ls_range`) rather than a default.

**Stage 5 — removing the optimiser as a variable closes the gap.**
PILCO's ``minimize`` is now ported to PyTorch (``pilco/optim/minimize.py``,
BFGS + cubic line search + Wolfe-Powell, ``length`` counted in line searches with
``MFEPLS`` evaluations each), and ``minimize_port_check.py`` validates it against
the NumPy original on the same objective from the same start: the analytic
quadratic traces match to 0.0, and on the real curbed GP marginal likelihood the
two converge to the *same* point (final value identical to 1e-12,
hyper-parameters to 1.4e-8; the traces differ mid-flight by 1.4e-6 because the
autograd gradient differs from the analytic one at the 1e-9 level, which can flip
a line-search decision).

Wiring it in as ``gp_optimizer="pilco"`` makes the fit match the reference
exactly::

    numpy  log ell[out0] = [5.1506 2.0613 5.9450 2.9242 3.6780]   -NLML -51.2716
    torch  log ell[out0] = [5.1506 2.0613 5.9450 2.9242 3.6780]   -NLML -51.2716
    hyper-parameter max abs diff = 0.0000

So the divergence really was the optimiser, and it is now a controlled variable
(``PilcoConfig.gp_optimizer`` / ``policy_optimizer``, both defaulting to
``"pilco"``; ``"lbfgs"`` selects ``torch.optim.LBFGS``).

For a fully faithful end-to-end comparison set ``PilcoConfig.legacy_rng=True`` as
well: that routes the process noise *and* the controller's initial random draws
through NumPy's legacy global stream (``np.random.seed`` / ``np.random.randn``),
matching ``util/gaussian.py`` and ``base/rollout.py`` draw for draw.  Without it
the two loops see different noise and their trajectories diverge for reasons that
have nothing to do with the maths.

```python
PilcoConfig(horizon=40, episodes=6, gp_iterations=150, policy_iterations=150,
            gp_optimizer="pilco", policy_optimizer="pilco", legacy_rng=True)
```

**Conclusion.**  The model layer is correct, the loop can learn, and the
remaining divergence is *a stopping rule*: PILCO's weak MATLAB-style `minimize`
halts at a smooth, slightly under-fitted GP, and at this data size that happens
to control better than a converged fit.  Reproducing it exactly would mean
imitating the stopping rule.  The open problem is therefore two-fold:
**long-run stability** of the model/policy co-adaptation, and **choosing the GP
fit by control quality rather than marginal likelihood**.  The collapse has the classic
PILCO shape: once the controller succeeds, the dataset concentrates near the
upright equilibrium, the GP hyper-parameters are re-fit to that narrower data,
and the policy optimiser — run for only 60 iterations — falls out of the
swing-up basin and cannot climb back.

**The three-step fix route is implemented** as `PilcoConfig` knobs
(`pilco/pilco.py`):

| step | knob | default | status |
| --- | --- | --- | --- |
| 1. optimiser budget | `policy_iterations` (L-BFGS iterations; PILCO counts *line searches* with `MFEPLS=30` each) | 300 | on |
| 1b. GP noise floor | `noise_floor` | 0.0 | **measured to hurt** — see the `fit_hyperparameters` docstring; left off |
| 2. bound model reshaping | `hyperparameter_trust_region` (log units per episode) | 0.0 | off by default |
| 2b. keep the best model | `keep_best_hyperparameters` | True | on |
| 3. never roll out a worse controller | `accept_if_improved` | True | on |

Step 3 works as follows: after the GP is re-fit, the *incumbent* controller's
model-predicted cost is recorded under the new model; the optimised controller is
kept only if it improves on that baseline, otherwise the incumbent parameters are
restored.  Every episode logs `baseline_value` and `accepted`.

**Measured effect of step 3** (`followup_experiments.py --part B`, 8 episodes,
`gp_iterations=30, policy_iterations=60`):

| run | realised cost per episode |
| --- | --- |
| gating OFF | 35.6, 30.3, 11.7, 12.2, 40.0, 38.1, 39.9, 40.0 |
| gating ON | 35.6, 30.3, 11.7, 12.2, 40.0, 38.1, 39.9, 40.0 |

Identical, and `accepted = [1]*8` — **the guard never fires**.  The gating-OFF run
reproducing the previously recorded numbers exactly is a useful regression check
of the refactor, but step 3 as formulated does not fix the collapse.

The reason is structural: step 3 is a *model-based* guard, and it compares the two
controllers **under the degraded model**.  Once the refit has lost the swing-up
region, both the incumbent and the new controller look bad to it, and the
optimiser's controller is the better of the two *in that model*.  A guard that
cannot see the model degrade cannot protect against it.

That motivated step 2c, `reject_worse_model`, which compares the **candidate fit
against the best-so-far fit on the same controller** and snaps back (with a short
re-fit, since a hard restore pairs old hyper-parameters with new data and was
observed to make the Gram matrix singular).  Measured:

| run | realised cost per episode (6 episodes) |
| --- | --- |
| unguarded | 35.6, 30.3, 11.7, 12.2, 40.0, 38.1 |
| `reject_worse_model` | 35.6, 26.4, 33.1, **9.7**, 39.9, 40.0 |

It changes the trajectory and reaches **9.7**, below the reference's own floor —
so the loop can find a better controller than PILCO's — but it **still collapses**,
and on this seed the 6-episode total is worse (184.7 vs 167.9).

**Stage 6 — with the reference optimiser the collapse disappears.**
The stability guards were treating a symptom.  Running the loop with
``gp_optimizer="pilco"``, ``policy_optimizer="pilco"`` and ``legacy_rng=True``
(same protocol, `H=40`, `gp_iterations=150`, `policy_iterations=150`, seed 0):

| episode | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| realised cost | 37.9 | 28.0 | 14.5 | 14.0 | 13.8 | **12.9** |

Against the same protocol driven by ``torch.optim.LBFGS``, which learned to 11.7
and then collapsed to 40.0.  The collapse was **entirely** an artefact of the GP
refit landing in a different basin under a different optimiser; with the
reference's own ``minimize`` the loop is stable and monotone.

Across seeds (`results/cartpole_faithful_seeds.json`):

| seed | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 37.9 | 28.0 | 14.5 | 14.0 | 13.8 | 12.9 |
| 1 | 38.3 | 35.1 | 33.3 | 30.2 | 26.9 | 19.5 |
| 2 | 34.8 | 29.3 | 14.4 | 12.7 | 12.0 | **11.6** |
| NumPy reference (faithful protocol) | 76.2 | 12.6 | 8.6 | 8.4 | 12.8 | — |

So: every seed learns monotonically and **none collapses**; seed 2 lands inside
the reference's own 8.3-12.8 band; seed 1 is slower and still descending at
episode 6.  Episode 0 differs from the reference (34.8-38.3 vs 76.2) because the
initial random rollout and controller draw do not match the reference's seeding
exactly, and that difference cascades through the closed loop.  The remaining
work is therefore *initialisation parity*, not stability or correctness.

**Verdict on the three-step route: implemented, measured, and superseded.**
`results/followup_guards_summary.json` collects the numbers:

| guard | measured verdict |
| --- | --- |
| acceptance (step 3) | inert — never fires |
| noise floor | hurts (39.1 / 39.3 / 35.3 vs 33.85) |
| shared lengthscales | hurts (38.3) |
| lengthscale curb `ls_range=3.0` | recovers about half the *reproduction* gap (34.91) |
| model rejection (step 2c) | changes the trajectory, reaches 9.7, still collapses |
| **reference optimiser (`gp_optimizer="pilco"`)** | **fixes it — no collapse** |

The guards are all kept as options but none is needed once the optimiser matches;
they are best read as evidence about *why* the loop was collapsing rather than as
recommended settings.

The structural reason all model-based guards fail is the same in each case: **the
model is the oracle the guard consults.** Once the refit has lost the swing-up
region, every controller — including the good one — looks bad to it, so no
comparison *within that model* can recover the right answer.

The remaining lever is therefore at the data/model level, not the guard level:
the loop already observes the realised cost of every episode it executes, so it
can validate the model on data it actually collected (compare the predicted cost
of the executed controller against the realised one and reject a refit whose
prediction error jumps), or detect covariate shift in the GP inputs when the
dataset concentrates near the equilibrium.  Neither is implemented here.


With ≤ 80 data points and 32 GP hyper-parameters (6 ARD lengthscales × 4 outputs,
plus 4 signal and 4 noise parameters) more optimisation is *worse*.  PILCO's
weaker MATLAB-style `minimize` accidentally regularises; the faithfully
implemented curb is a sharp `p = 30` penalty that only bites at ~100× deviations
and is negligible in the relevant range.

**Why this is a documentation-worthy state rather than a footnote.**  It is the
same failure mode as the directed-exploration problem: the closed-loop result is
dominated by how much the *model* is trusted, not by the planner's algebra.  Any
DE comparison run on top of an overconfident dynamics model will be measuring
that, not the exploration objective.

**Next steps.**  (1) Match PILCO's `minimize` convergence behaviour (bounded
line searches or NLML-based early stopping) rather than maximising the marginal
likelihood; (2) reduce the hyper-parameter count per output until data ≫
parameters; (3) instrument the closed loop episode-by-episode against the
reference on a shared dataset.
