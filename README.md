# `pilco_pytorch`

A PyTorch re-implementation of PILCO (Deisenroth & Rasmussen 2011) in which
every nonlinearity and every Gaussian moment propagation is a small,
composable, autograd-differentiable module — plus closed-form directed
exploration objectives on top of it.

The design argument is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the
directed-exploration design (and the diagnosis of why it does not help on
cart-pole) is in [`docs/DIRECTED_EXPLORATION.md`](docs/DIRECTED_EXPLORATION.md)
and in the repository-level [`reports/ANALYSIS.md`](../reports/ANALYSIS.md).

---

## The idea in one paragraph

PILCO's analytic machinery is one pattern repeated: given `x ~ N(m, S)`, report
`E[g(x)]`, `Cov[g(x)]` and `C = S⁻¹ Cov[x, g(x)]`.  The NumPy port
(`../pilco_python`) implements that pattern *and* hand-derives all nine
Jacobians (`dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp, dSdp, dCdp`) for every
operator, in MATLAB's column-major `vec` convention — over half the code base,
and the source of most bugs.  Here an operator only implements the **forward**
moments; autograd produces every derivative, including second-order ones.

```python
class MomentOp(nn.Module):
    n_in: int
    n_out: int
    def moments(self, mean, cov) -> Moments: ...   # E[g], Cov[g], C
```

`splice()` then attaches an operator's output to an augmented Gaussian in ten
lines — replacing `base/propagated.py::_fill_in` and its Fortran index
bookkeeping.

---

## Quick start

```bash
pip install -r requirements.txt

# validate the operators against the NumPy reference (machine precision)
python tests/test_ops_vs_numpy.py
python tests/test_joint.py

# validate the whole planner + gradients against the NumPy reference
python tests/test_end_to_end_vs_numpy.py

# learn cart-pole
python scripts/run_cartpole.py --episodes 6 --seed 0

# compare PILCO against the closed-form exploration objectives
python scripts/benchmark_exploration.py --env cartpole --episodes 5 --seeds 0 1 2
python scripts/benchmark_exploration.py --env mountaincar --episodes 4 --horizon 120
```

Library use:

```python
import torch
from pilco.scenarios.cartpole import make_cartpole
from pilco.pilco import PilcoConfig, learn
from pilco.objectives import ExplorationObjective

torch.set_default_dtype(torch.float64)
scenario = make_cartpole()
config = PilcoConfig(
    horizon=40, episodes=6, seed=0,
    objective=ExplorationObjective(kind="gittins"))   # or "ucb", "mean"
model, logs = learn(scenario, config)
```

---

## Layout

```
pilco/
  gaussian.py    Gaussian container + safe linear algebra
  ops/           moment operators
    base.py        MomentOp, Moments, splice, AffineOp
    fourier.py     sin/cos harmonics  (replaces gTrig + gSin + gSat)
    policy.py      RBF controller     (replaces congp + gp2)
    gp.py          ARD SE GP dynamics (replaces gp0d)
    cost.py        saturating cost    (replaces lossSat / loss_cp)
    joint.py       joint trig augmentation of two correlated states
  trajectory.py  StateSpec, PilcoModel, plan()
  objectives.py  value, cumulative-cost variance, information gain, acquisitions
  models/gp.py   L-BFGS hyper-parameter training (replaces train + hypCurb)
  pilco.py       the learning loop
  scenarios/     cartpole.py, mountaincar.py
```

## Validation

| check | result |
| --- | --- |
| `TrigAugmentOp` vs `util/gTrig.py` | exact (0, 6e-17, 0) |
| `SaturationOp` vs `util/gSat.py` | exact (6e-17, 4e-16, 6e-17) |
| `GpDynamicsOp` vs `gp/gp0d.py` | 3e-11 / 8e-12 / 3e-11 |
| `SaturatingCostOp` vs `loss/lossSat.py` | 0 / 1.7e-18 |
| cost cross-moment vs Monte Carlo | 1.7e-4 |
| autograd vs finite differences | 3e-10 |
| `plan().value()` vs `base/value.py` | 1.2e-6 absolute on 39.8 |
| autograd gradient vs `base/value.py` analytic `dJ/dp` | 5.6e-6 relative |
| mean Jacobian (`jacfwd` vs `jvp`) | 4.6e-13 |
| `augmented_joint` vs MC (in standard-error units) | ≤ 3.6σ over 3e6 samples |
| cumulative-cost variance vs joint-Gaussian MC | 4.8e-2 relative |

## Status and a known limitation

**The model layer is done and validated; the learning loop is not yet at parity.**

* All operators, the planner, the value and its gradient match the NumPy
  reference (table above).  On *identical* data and an identical starting policy
  the policy optimiser is comparable too: NumPy `minimize` reaches `J = 32.37`
  in 151 line searches, torch L-BFGS reaches `J = 33.34` in 187 evaluations.
* The closed loop **does learn** — running eight episodes with
  `gp_iterations=30, policy_iterations=60` reaches **11.7 and 12.2** per
  `H = 40` episode at episodes 3–4, against the NumPy reference's floor of
  8.3–12.8 (see `results/cartpole_learning_gp30.json`).  It then **collapses back
  to the cost ceiling** (40.0) and stays there.  So the open problem is long-run
  *stability* of the model/policy co-adaptation, not the ability to learn.
* **Where the two implementations actually diverge** is now pinned down.  They
  agree on the initial dataset, on the value (7e-7), on the gradient at *every*
  one of the 57 objective evaluations PILCO's optimiser makes (max 1.5e-4 relative),
  and on the GP
  training objective (1e-9 per output).  The divergence is the *point the GP fit
  stops at*: NumPy's `minimize` halts at a much smoother model
  (log-lengthscales up to 5.95) than a converged fit (2.73), and on 40 data
  points the smoother model controls better (policy value 33.85 vs 35.80).
  Autograd drift is ruled out.  **PILCO's own `minimize` is now ported to
  PyTorch** (`pilco/optim/minimize.py`) and used by default
  (`PilcoConfig.gp_optimizer = policy_optimizer = "pilco"`); with it the GP fit
  matches the reference *exactly* (hyper-parameter max abs diff 0.0000, −NLML
  both −51.2716), which confirms the optimiser was the whole story.
* **The collapse is explained and fixed.**  It was not a stability problem at
  all: with `torch.optim.LBFGS` the GP refit lands in a different basin and the
  loop loses the policy.  Using the ported `minimize`
  (`gp_optimizer="pilco", policy_optimizer="pilco", legacy_rng=True`) the loop is
  stable and monotone across seeds.  Over **12 episodes** seed 0 converges and
  holds: `37.9 28.0 14.5 14.0 13.8 12.9 12.9 12.8 12.8 12.9 12.8 12.7`; seed 2
  reaches `10.4` at episode 7 then settles at ~16.7; seed 1 is slower.  Against
  11.7 then 40.0 for the L-BFGS run.  **No seed collapses.**  Episode 0 differs
  from the reference because the initial random rollout does not match the
  reference's seeding exactly, and that cascades; the remaining work is
  initialisation parity, not stability.
* **Profile** (`scripts/diagnostics/profile_hotspots.py`): the one-step moment map
  is ~100% of a planning step (12.4 ms, of which the GP moments are 13.7 ms), and
  `jacfwd` costs `x5.46` on top, which is why non-variance objectives now skip it.
  The `information_gain` objective was **444.8 ms and is now 1.47 ms (~300x)**
  after batching `expected_kernel_matrix` over the query pairs.
  `cumulative_cost_variance` is the remaining hotspot at 706 ms (122% of a plain
  plan) and is the obvious next target.
* The four stability guards I added are therefore **superseded**: the acceptance
  guard never fires, the noise floor and shared lengthscales hurt, the
  lengthscale curb recovers about half the *reproduction* gap, and the
  model-rejection guard reaches 9.7 but still collapsed.  They are kept as
  options and as evidence about the failure mechanism.

Diagnosis so far (`scripts/diagnostics/optimiser_parity.py` and the experiments
in `docs/ARCHITECTURE.md` §8): the divergence is in **GP hyper-parameter
training**, not in the planner.

* On the same 40-point dataset, the NumPy trainer stops at a negative log
  marginal likelihood of `+51.3`, while unconstrained L-BFGS reaches `−25.4`
  from the same starting point (it drives the noise standard deviation to
  ~1e-4, i.e. it fits the sampled process noise).
* That overconfident model then misleads policy search.  Measured closed-loop
  effect on cart-pole: **30 GP iterations → episode costs `[35.0, 36.3, 19.2]`**
  (learning), **150 iterations → `[40.0, 37.7, 29.4]`** (not learning).
  With ≤ 80 data points and 32 GP hyper-parameters, more optimisation is worse.
  PILCO's weak MATLAB-style `minimize` accidentally regularises; the faithful
  curb is a sharp penalty at ~100x deviations and does not bite in the relevant
  range.
* The default `gp_iterations` is therefore set to 50 with this reasoning
  documented in `pilco.py`, but that is a workaround, not a fix.

Concrete next steps, in order: (1) match PILCO's `minimize` convergence more
closely (bounded line searches / early stopping on the NLML, which is the
behaviour the reference actually exhibits); (2) reduce the hyper-parameter count
per output or share lengthscales until there are ≫ data points; (3) instrument
the closed loop episode-by-episode against the reference on a *shared* dataset.

## Directed exploration

`objectives.ExplorationObjective` selects among

* `mean` — plain PILCO,
* `ucb` — `E[C] − β·sd[C]`,
* `gittins` — `E[C] + λ'_γ·sd[C]`, `λ'` from McAllister eq. (4.16),
* `info_weight > 0` — adds `−w ·` closed-form GP information gain
  (`0.5 Σ_e log det(I + σ_e⁻² K̄_e)`, normalised so that it measures the
  non-redundancy of the visited inputs).

Unlike `pilco_python`, the exploration term is **differentiated** and optimised
by the same L-BFGS call as the cost.  See `docs/DIRECTED_EXPLORATION.md`.

## Requirements

Python ≥ 3.10, `torch ≥ 2.0` (for `torch.func`), `numpy`, `scipy` (only for the
true-environment ODE in the cart-pole scenario).  The reference-validation tests
additionally need `../pilco_python` on the path.
