# Closed-form directed exploration in the PILCO framework

This document is the technical companion to `reports/ANALYSIS.md`.  It states
what directed exploration is supposed to do, which parts of it stay closed form
under PILCO's moment matching, what is implemented in `pilco/objectives.py`, and
under which conditions the objective can actually beat plain PILCO.

---

## 1. Source correction

The analysis document shipped with the repository
(`directed_exploration_analysis.md`) attributes the full cumulative-cost
distribution and the Bayesian-optimisation acquisitions (UCB/PI/EI/Gittins) to
*Deisenroth (2015), Chapter 4*.  That attribution is wrong:

| claim | actual source |
| --- | --- |
| `V(x₀) = Σ_t E[c(x_t)] + b·σ_{x_t}[c(x_t)]` (marginal cost std, `b ∈ [-0.5, 0]`) | Deisenroth thesis §3.7.1 eq. (3.50) — the only directed exploration in *his* thesis |
| full joint `C_ψ ~ N(μ^C_e, Σ^C_e)` and UCB/PI/EI/Gittins over policy parameters | McAllister, *Bayesian Learning for Data-Efficient Control*, Cambridge 2017, **Ch. 4** |
| "cross terms contribute 40–85% of the cumulative-cost variance" | McAllister Ch. 4 §4.2 |
| `GI = μ_C + λ'_γ σ_C`, `g(λ') = γ(φ(λ') + λ'Φ(λ') − λ') + λ' = 0` | McAllister eqs. (4.15)–(4.21) |

Two consequences matter for the diagnosis in `reports/ANALYSIS.md`:

* `λ' < 0`, so `GI = μ_C + λ'σ_C` is a **lower** confidence bound (optimistic
  under cost minimisation), not "UCB with an adaptive β".  The file
  `base/directed_explore.py` implements it correctly; the comment calling it an
  extension of UCB is wrong.
* McAllister's own headline result on cart-double-pole is
  `PILCO 18.82`, `GI 18.58`, `EI 19.00`, `PI 19.11`, `UCB(0.3) 18.97` with
  standard errors of about `0.5` — i.e. **the method's author could not show a
  statistically significant win over PILCO** on the hardest standard benchmark.
  Expecting a robust win on the much easier cart-pole is unrealistic.

---

## 2. What closed-form means here

PILCO never sees the true dynamics; it propagates a Gaussian through the GP's
analytic predictive moments.  The planner therefore needs, for each
nonlinearity, the three quantities of §1 of `ARCHITECTURE.md`.  Along a planned
rollout the state distribution at each step is Gaussian, so any objective that
is a *smooth function of the per-step means and covariances* is differentiable
end to end and can be optimised with L-BFGS.  This is the design space:

| objective | closed form? | what it measures | implemented |
| --- | --- | --- | --- |
| `Σ_t E[c(x_t)]` | yes (PILCO) | exploitation | `expected_cumulative_cost` |
| `Σ_t Var[c(x_t)]` | yes | *marginal* outcome uncertainty | `traj.cost_vars` |
| `Var[Σ_t c(x_t)]` incl. cross terms | yes, via Jacobian chain | total outcome uncertainty (McAllister Σ^C_e) | `cumulative_cost_variance` |
| `0.5 Σ_e log det(I + σ_e⁻² K̄_e)` | yes | **model improvement** (GP mutual information / D-optimality) | `information_gain` |
| `Σ_t E[σ_f²(x_t)]` | yes (linear in `Q`) | predictive variance (V-optimal) | not implemented (undirected; see §5) |
| `E[log det(I + σ⁻² K̄)]` | **no** — only `log det` of the *expected* matrix | — | Jensen-biased approximation |

The first four are what the benchmark switches between.

### 2.1 Cumulative-cost variance

```
Σ^C_e = Σ_t Σ_t' Cov[c(X_t), c(X_t')]                     (McAllister 4.24)
Cov[X_a, X_b] = S_a J_aᵀ J_{a+1}ᵀ … J_{b-1}ᵀ,  J_i = ∂m_{i+1}/∂m_i    (4.27)
Cov[c(X_1), c(X_2)] = E[c(X_1)c(X_2)] − E[c(X_1)]E[c(X_2)]            (4.35)
```

`cost_cross_covariances` builds the first, `SaturatingCostOp.cross_moment` the
third (on the trig-augmented joint, via `ops/joint.py`), and `J_i` comes from
`torch.func.jacfwd`.  The NumPy reference implements the same three pieces but,
per the audit in `reports/ANALYSIS.md`, with a transposed/permuted Jacobian chain
and a wrong marginal-variance formula, giving ~48% error on a real cart-pole
trajectory.

### 2.2 Information gain — the genuinely new piece

For a GP with kernel `k_e` and noise `σ_e`, the mutual information between the
observations at query points `z_1..z_n` and the latent function is

```
I(y; f) = 0.5 Σ_e log det(I + σ_e⁻² K_e)                  (Srinivas et al. 2010)
```

which is the GP-UCB / D-optimal design criterion.  The obstacle is that the
query points are *random*: the trajectory is a Gaussian sequence, so the honest
objective is `E[log det(·)]`, which has no closed form.

The approximation taken here is the standard one (and the one Scannell et al.
2023 use for their closed-form entropy objective): replace `K` by its
expectation `K̄_e[t,s] = E[k_e(z_t, z_s)]`.  That expectation **is** closed form
for the squared-exponential kernel:

```
K̄_e[t,s] = sf_e² |I + Λ V_t|^{-1/2} exp(−½ δᵀ Λ (I + V_s Λ)^{-1} δ)
δ = μ_t − μ_s,   V = Cov(z_t) + Cov(z_s) − Cov(z_t,z_s) − Cov(z_t,z_s)ᵀ
```

Because the diagonal is exactly `sf_e²` (independent of the policy), the raw
criterion is dominated by a constant; `information_gain(normalized=True)`
subtracts it and divides by the number of query points, so the objective measures
the **non-redundancy of the visited inputs** in the GP's own metric.  It is a
smooth, differentiable function of the policy parameters and is optimised by the
same L-BFGS call as the cost — which is precisely the "closed-form directed
exploration" idea, as opposed to the Monte-Carlo candidate search in
`pilco_python/scenarios/cartPole/benchmark_de.py`.

Known biases, stated honestly:

* the off-diagonal expectation treats `z_t` and `z_s` as independent, which
  *under*-estimates redundancy and therefore over-values exploration; the exact
  joint would use `Cov(z_t, z_s)` and is a drop-in change
  (`expected_kernel_matrix(..., cross_covs=...)`);
* plugging in `E[K]` inside a `log det` is a Jensen approximation;
* the criterion is cost-blind: it rewards covering distinct inputs, not distinct
  *relevant* inputs.  It must be combined with the cost term to stay on-task.

---

## 3. Why directed exploration cannot beat PILCO on cart-pole

Measured, not asserted (`pilco_python/tmp/exp_headroom.py`; see
`reports/ANALYSIS.md` §2 for the full tables):

* **Faithful protocol** (an `H = 40` step random initial rollout, 150 BFGS
  iterations, 5 episodes): episode 0 costs ≈ 74/80, and episodes 1–4 cost
  **8.3–12.8/80**.  PILCO reaches the floor after a single learning episode.
* **The repository's benchmark protocol** (12-step initial rollout, 10 BFGS
  iterations, 8 episodes): PILCO *never learns*.  Cost per episode stays at
  46–80 out of 80, i.e. the controller is as bad as a random one.  Every
  reported "improvement" is the difference between two equally failing policies,
  and the across-seed spread (`+25%` to `−6%` in the logs) is pure noise.

So there are only two regimes, and neither can show an exploration win:

1. **PILCO already at the floor** → the exploration bonus can only add cost.
   This is a *no-headroom* problem, not an "environment too simple" problem in
   the usual sense: cart-pole is easy *for PILCO*, and the total-cost metric has
   nothing left to buy.
2. **PILCO crippled** → both methods are at the cost ceiling; differences are
   noise.

Add to that three implementation-level reasons the NumPy DE arm was weak
independently of the environment:

* the analytic cumulative variance that the theory needs was **dead code**:
  `base/value_de.py` uses a sum of *marginal* per-step variances, which ignores
  exactly the cross-timestep terms the theory says dominate;
* `base/analytical_variance.py` is itself wrong (~48% error, audit in
  `reports/ANALYSIS.md` §4);
* the exploration objective is never differentiated: `benchmark_de.py` optimises
  `μ` with BFGS and then picks the best of 15 *random* parameter perturbations
  by BO score.  That is undirected search with an exploration-flavoured ranking,
  so it cannot be more efficient than PILCO.

---

## 4. Necessary conditions for directed exploration to pay off

From the above, a test bed is only informative if **all** of the following hold.

1. **Cost headroom within the planning horizon.**  PILCO must be visibly above
   the cost floor.  Quantify it: `(greedy cost − best achievable cost) / greedy
   cost`.  Cart-pole's is ≈ 0 after one episode.
2. **The bottleneck is model error, not the policy optimiser.**  If the GP is
   already accurate on the states the greedy policy visits, extra data is worth
   nothing.  Measure the GP's predictive variance along the planned trajectory:
   it must stay O(1) in the kernel's units.
3. **The informative behaviour pays off inside the horizon.**  The DE objective
   is the same finite-horizon cost with an uncertainty bonus; if reaching the
   informative region takes longer than `H`, the bonus cannot be recovered.  This
   is the crux for mountain car (see §6).
4. **Exploration must be gradient-optimisable.**  The score has to be a
   differentiable function of the policy parameters, so that L-BFGS can *seek*
   informative trajectories rather than sample them.  This is what
   `objectives.ExplorationObjective` provides.
5. **Enough episodes for the information to compound.**  The Gittins index's
   value of information grows with the remaining episode budget; with 2–3
   episodes it is ≈ 0 by construction.

Cart-pole fails (1).  Cart-double-pole fails (2) in the *opposite* direction:
both arms of McAllister's comparison sit at the cost ceiling, which is why the
published differences are within one standard error.

---

## 5. Why the sum-of-marginal-variances proxy is not enough

`pilco_python/base/value_de.py::_compute_marginal_variance` returns
`Σ_t Var[c(X_t)]`.  Even computed exactly (it is not — the audit shows ~48%
error), this is the right quantity only if the per-step costs are uncorrelated.
Along a trajectory they are strongly positively correlated, so the true
`Var[Σ c_t]` is much larger: on the cart-pole trajectory used in
`tests/test_end_to_end_vs_numpy.py` the marginal sum is `0.046` with the exact
Gaussian-model total `0.214`.  A bonus proportional to `√Σ_t Var[c_t]` therefore
under-explores in exactly the way the theory warns about.

That test also quantifies the *model* error: evaluating the cost on the true
`[x, sin θ, cos θ]` (sampled exactly) rather than on its Gaussian image gives
`0.114` instead of `0.214`.  So even a correct closed-form `Σ^C_e` inherits a
~2× calibration error from PILCO's moment closure.  Any DE objective based on
cost variance should be expected to be correspondingly noisy — an argument for
the information-gain objective, which is defined directly in the GP's own metric
and does not rely on the cost's Gaussian image.

---

## 6. Recommended test bed: mountain car

`pilco/scenarios/mountaincar.py` implements Gym's continuous mountain car with a
saturating cost on the distance to the goal.  Why it is the right target:

* **Structure.**  A myopic "push towards the goal" policy parks the car at
  `x ≈ −0.08` and never reaches the goal at `x = 0.45`; an energy-pumping
  strategy reaches it.  This is the canonical informational failure — the greedy
  policy is not merely suboptimal, it *never generates the data* that would
  reveal the better policy.
* **Closed form preserved.**  The state is 2-D, the dynamics are smooth apart
  from the velocity/position clamps, and the cost is a saturating quadratic, so
  every Gaussian moment in the planner is already closed form.  No new operator
  is needed.
* **Measured headroom.**  With the default `time_scale=8` (force, gravity and
  `v_max` scaled together, which shortens the pumping time without changing the
  difficulty) and `horizon=120`, over the planning horizon:

  | policy | mean cost per step over `H = 120` |
  | --- | --- |
  | greedy "always push right" | 0.918 |
  | energy pumping (`a = sign(v)`) | 0.586 |

  a **36% gap**, versus ≈ 0% for cart-pole after one episode.
* **Novelty.**  The literature survey found no published PILCO-style directed
  exploration result on mountain car.

**The horizon requirement is not incidental.**  With the *unscaled* dynamics
(`time_scale=1`) the pumping strategy reaches the goal at step 169; if `H = 80`,
the greedy policy has *lower* cumulative cost than the pump inside the horizon
and directed exploration cannot pay off.  This is condition (3) of §4 in its
purest form, and it is why mountain-car benchmarks must be reported with an
explicit horizon argument.

---

## 7. What would make the theory airtight

The genuinely open piece (nobody appears to have published it) is a fully
derived `∂Σ^C_e/∂ψ`.  In this code base that derivative is free — the
`cumulative_cost_variance` objective is differentiable and L-BFGS uses its
gradient — so the remaining gaps are conceptual rather than algebraic:

1. Replace the independent off-diagonal approximation in `expected_kernel_matrix`
   with the exact joint `Cov(z_t, z_s)` (available from the trajectory's
   cross-covariances).
2. Weight the information gain by cost relevance, e.g.
   `Σ_t w(c_t) · [info gain at step t]`, so that exploration is directed towards
   *consequential* uncertainty (McAllister's stated criticism of undirected
   design criteria).
3. Correct the aleatoric/epistemic confusion: PILCO's predictive variance
   includes the GP observation noise, which cannot be reduced by exploration.
   McAllister §4.5 sketches a rank-1 fix; the general T-rank version is open.
4. Re-calibrate the cost-variance path, or use the information-gain path, given
   the ~2× moment-closure error measured in §5.
