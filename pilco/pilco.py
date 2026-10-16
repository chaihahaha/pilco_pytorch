"""The PILCO learning loop, with pluggable directed exploration.

The loop is the textbook one:

1. roll out the current controller on the *true* system and append the data;
2. re-fit the dynamics GP (warm-started hyper-parameters);
3. optimise the controller parameters against an
   :class:`~pilco.objectives.ExplorationObjective`;
4. repeat.

Everything is differentiable through the closed-form moment propagation, so
step 3 is a plain L-BFGS call -- including when the objective contains the
analytic cumulative-cost variance or the closed-form information gain.  The
NumPy implementation instead runs PILCO's gradient-based optimiser on the mean
only and then does a random-perturbation candidate search for the exploration
term, discarding the analytic exploration gradient entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .gaussian import Gaussian
from .models.gp import (build_dynamics_op, fit_hyperparameters,
                       fit_hyperparameters_pilco, initial_log_hyp)
from .objectives import ExplorationObjective
from .ops.policy import RbfPolicyOp
from .trajectory import PilcoModel, plan

__all__ = ["PilcoConfig", "initial_policy", "optimise_policy", "learn",
           "EpisodeLog"]

DTYPE = torch.float64

#: L-BFGS iterations used when snapping the GP back towards the best-so-far fit.
SNAP_BACK_ITERS = 15


@dataclass
class PilcoConfig:
    """Hyper-parameters of the learning loop."""

    horizon: int = 40
    episodes: int = 6
    initial_rollouts: int = 1
    #: L-BFGS iterations for the GP hyper-parameters.  This is *not* a PILCO
    #: parameter: PILCO's ``minimize`` runs 150 MATLAB-style line searches, which
    #: does far less work than 150 strong-Wolfe L-BFGS iterations.  With only 40
    #: data points and 32 hyper-parameters (6 lengthscales x 4 outputs, plus
    #: signals and noises) aggressive optimisation fits the sampled process noise
    #: and the resulting overconfident model misleads policy search.  Measured on
    #: cart-pole: 30 iterations -> episode costs [35.0, 36.3, 19.2] (learning),
    #: 150 iterations -> [40.0, 37.7, 29.4].  See docs/ARCHITECTURE.md section 8.
    gp_iterations: int = 50
    policy_iterations: int = 300
    objective: ExplorationObjective = field(default_factory=ExplorationObjective)
    seed: int = 0
    verbose: bool = True
    #: re-fit hyper-parameters from scratch every episode (PILCO warm-starts)
    retrain_from_scratch: bool = False

    # ------------------------------------------------------------------
    # Stability guards for the model/policy co-adaptation.  These are what the
    # experiments in docs/ARCHITECTURE.md section 8 identified as the remaining
    # problem: the loop learns cart-pole and then loses it again.  The three
    # knobs below correspond to the three-step fix route.
    # ------------------------------------------------------------------
    #: Step 1 -- lower bound on the GP observation-noise standard deviation.
    #: The GP targets are state differences from a noisy rollout, so their noise
    #: has a physical floor.  Without one the marginal-likelihood maximiser
    #: drives the noise to ~1e-4, the model becomes overconfident, and the
    #: planner stops correcting.  Set this to the plant's process-noise std.
    noise_floor: float = 0.0
    #: Step 2 -- trust region (in log units) on how far one episode may move the
    #: GP hyper-parameters from the previous episode's values.  ``0`` disables
    #: it.  This bounds how much a single episode's data can reshape the model.
    hyperparameter_trust_region: float = 0.0
    #: Step 2b -- keep the hyper-parameters that scored the best *model-predicted*
    #: value so far instead of blindly accepting the newest fit.  Restoring
    #: happens once, at the end of learning, so the returned model is the best
    #: model-predicted one.
    keep_best_hyperparameters: bool = True
    #: Step 2c -- reject a GP *refit* that makes the known-good controller look
    #: worse.  This targets the measured failure mechanism: the episode-3/4
    #: controller solves cart-pole, then the next refit (on data now concentrated
    #: near the equilibrium) both loses the swing-up region and reports the
    #: incumbent controller as bad, so the planner is talked out of it.  The
    #: step-3 guard cannot catch that, because it compares controllers *under the
    #: degraded model*.  Here the candidate fit and the best-so-far fit are
    #: compared on the same controller: if the best-so-far model gives it a lower
    #: predicted cost, the refit is discarded.
    reject_worse_model: bool = False
    #: Step 3 -- accept the newly optimised controller only if it improves the
    #: model-predicted cost under the current model (otherwise keep the previous
    #: one).  This is the "never roll out a worse controller" guard.
    accept_if_improved: bool = True
    #: draw process noise from NumPy's legacy global stream
    #: (``np.random.seed`` / ``np.random.randn``) exactly like
    #: ``pilco_python`` does.  Only useful for an apples-to-apples
    #: reproduction comparison.
    legacy_rng: bool = False
    #: Step 1c -- lengthscale-curb half-width, in log units, relative to
    #: ``log std(inputs)``.  ``None`` means PILCO's own curb (``log 100``,
    #: which is loose enough to be inactive in practice).  Measured effect
    #: on the 40-point cart-pole dataset, policy value after the same
    #: optimiser: reference fit 33.85, no curb 35.80, 3.0 -> 34.91,
    #: 2.0 -> 35.71, 1.5 -> 36.01.  ``3.0`` therefore recovers about half
    #: of the gap, but the response is not monotonic, so it is an option
    #: rather than a default (see docs/ARCHITECTURE.md section 8).
    ls_range: Optional[float] = None
    #: Which optimiser to use.  ``"pilco"`` is the PyTorch port of
    #: PILCO's MATLAB-style ``minimize`` (``pilco/optim/minimize.py``);
    #: ``"lbfgs"`` is ``torch.optim.LBFGS`` with a strong-Wolfe line search.
    #: They are NOT interchangeable: counted in line searches with
    #: ``MFEPLS`` evaluations each, PILCO's ``minimize`` does far less work
    #: than ``LBFGS(max_iter=N)``, and on small datasets that difference
    #: decides whether the GP stays smooth enough for the planner to trust.
    #: Reproducing ``pilco_python`` requires ``"pilco"``.
    gp_optimizer: str = "pilco"
    policy_optimizer: str = "pilco"
    #: Function evaluations allowed per line search (PILCO's ``MFEPLS``).
    mfephls: int = 30


@dataclass
class EpisodeLog:
    episode: int
    realised_costs: np.ndarray
    predicted_value: float
    predicted_sd: float
    information_gain: float
    n_data: int
    policy_iterations: int
    #: model-predicted value of the controller *before* this episode's
    #: optimisation, under this episode's model (the acceptance baseline)
    baseline_value: float = float("nan")
    #: whether the optimised controller was accepted by the step-3 guard
    accepted: bool = True


def initial_policy(scenario, initial_state: Gaussian, n_basis: int,
                   generator: torch.Generator) -> RbfPolicyOp:
    """Initialise the controller the way ``settings_cp.m`` does.

    PILCO draws the pseudo-inputs from the *prior state distribution pushed
    through the trig augmentation* and the pseudo-targets from
    ``0.1 * N(0, 1)``; the ARD lengthscales are fixed at ``[1, 1, 1, 0.7, 0.7]``
    with unit signal and ``0.01`` noise.
    """
    spec = scenario.spec
    from .ops.fourier import TrigAugmentOp

    d = spec.state_dim
    trig = TrigAugmentOp(spec.angle_indices, n_in=d)
    t = trig.moments(initial_state.mean, initial_state.cov)
    aug_mean = torch.cat([initial_state.mean, t.mean])
    proj = initial_state.cov @ t.cross
    aug_cov = torch.zeros(spec.aug_state_dim, spec.aug_state_dim, dtype=DTYPE)
    aug_cov[:d, :d] = initial_state.cov
    aug_cov[:d, d:] = proj
    aug_cov[d:, :d] = proj.transpose(0, 1)
    aug_cov[d:, d:] = t.cov

    poli = list(spec.policy_indices)
    mean = aug_mean[poli]
    cov = aug_cov[poli][:, poli]
    L = torch.linalg.cholesky(cov + 1e-10 * torch.eye(len(poli), dtype=DTYPE))
    z = torch.randn(len(poli), n_basis, dtype=DTYPE, generator=generator)
    centers = (mean[:, None] + L @ z).transpose(0, 1)          # (n_basis, d_poli)
    targets = 0.1 * torch.randn(n_basis, spec.control_dim, dtype=DTYPE,
                                generator=generator)
    default_ell = torch.tensor([1.0, 1.0, 1.0, 0.7, 0.7], dtype=DTYPE)
    ell = default_ell[:len(poli)] if len(poli) == default_ell.numel() \
        else torch.ones(len(poli), dtype=DTYPE)
    return RbfPolicyOp(centers, targets, ell, noise=0.01, signal=1.0)


def make_model(scenario, inputs: torch.Tensor, targets: torch.Tensor,
               policy: RbfPolicyOp, log_hyp: Optional[torch.Tensor] = None,
               control_noise: float = 0.0) -> PilcoModel:
    from .ops.cost import TrigAugmentedCostOp

    dynamics = build_dynamics_op(inputs, targets, log_hyp)
    return PilcoModel(scenario.spec, dynamics, policy, scenario.cost,
                      max_u=[scenario.max_u] * scenario.spec.control_dim,
                      control_noise=control_noise)


def optimise_policy(model: PilcoModel, initial_state: Gaussian, config: PilcoConfig,
                    objective: ExplorationObjective,
                    verbose: bool = False) -> tuple[float, float]:
    """Optimise the controller parameters.  Returns the objective value.

    Uses the same optimiser family as the GP fit (see
    :attr:`PilcoConfig.policy_optimizer`), so that a comparison against
    ``pilco_python`` does not mix optimiser differences with model differences.
    """
    params = [model.policy.centers, model.policy.targets,
              model.policy.log_lengthscales]
    gp_params = [model.dynamics.log_lengthscales, model.dynamics.log_signal,
                 model.dynamics.log_noise]
    for p in gp_params:
        p.requires_grad_(False)

    if config.policy_optimizer == "pilco":
        from .optim.minimize import ParamVector, autograd_objective, minimize

        vec = ParamVector(params)
        evaluate = autograd_objective(
            lambda: objective(plan(model, initial_state, config.horizon,
                                   with_jacobians=True), model), vec)
        res = minimize(vec.flat(), evaluate,
                       length=config.policy_iterations,
                       MFEPLS=config.mfephls)
        vec.set(res.x)
        for p in gp_params:
            p.requires_grad_(True)
        if verbose:
            print(f"    policy: {len(res.fX)} line searches, "
                  f"objective {res.final:.4f}")
        return res.final, len(res.fX)

    optimiser = torch.optim.LBFGS(params, lr=1.0,
                                  max_iter=config.policy_iterations,
                                  tolerance_grad=1e-7, tolerance_change=1e-11,
                                  line_search_fn="strong_wolfe")
    state = {"value": float("inf"), "n": 0}

    def closure():
        optimiser.zero_grad(set_to_none=True)
        traj = plan(model, initial_state, config.horizon, with_jacobians=True)
        score = objective(traj, model)
        score.backward()
        # autograd can leave a non-contiguous .grad (e.g. from a transpose in the
        # moment formulas), which torch.optim.LBFGS rejects.  Detach as well: a
        # graph-attached .grad left behind on the parameter makes the *next*
        # backward pass fail with "backward through the graph a second time".
        for p in params:
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.detach().contiguous()
        state["value"] = float(score.detach())
        state["n"] += 1
        return score

    optimiser.step(closure)
    for p in gp_params:
        p.requires_grad_(True)
    if verbose:
        print(f"    policy: {state['n']} L-BFGS evaluations, "
              f"objective {state['value']:.4f}")
    return state["value"], state["n"]


def learn(scenario, config: Optional[PilcoConfig] = None,
          log: Optional[Callable[[str], None]] = print):
    """Run the full PILCO learning loop on ``scenario``.

    Returns ``(model, logs)`` where ``model`` is the final
    :class:`~pilco.trajectory.PilcoModel` and ``logs`` a list of
    :class:`EpisodeLog`.
    """
    config = config or PilcoConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # route the scenario's process noise through NumPy's legacy stream so that a
    # reproduction comparison against pilco_python sees the SAME noise
    if hasattr(scenario, "use_legacy_rng"):
        scenario.use_legacy_rng = bool(config.legacy_rng)
    generator = torch.Generator().manual_seed(config.seed)
    rng = np.random.default_rng(config.seed + 1000)

    initial_state = scenario.initial_state()
    plan_state = scenario.optimisation_state()

    # --- initial random rollouts -------------------------------------------
    inputs_np, targets_np = [], []
    for _ in range(config.initial_rollouts):
        model = None
        x0 = (initial_state.mean.numpy()
              + np.linalg.cholesky(initial_state.cov.numpy())
              @ rng.standard_normal(initial_state.dim))
        states, controls, _ = _random_rollout(scenario, x0, config.horizon, rng)
        i, t = scenario.data_from_rollout(states, controls)
        inputs_np.append(i)
        targets_np.append(t)

    inputs = torch.as_tensor(np.vstack(inputs_np), dtype=DTYPE)
    targets = torch.as_tensor(np.vstack(targets_np), dtype=DTYPE)

    policy = initial_policy(scenario, initial_state, scenario.n_basis, generator)
    log_hyp = initial_log_hyp(inputs, targets)

    logs: list[EpisodeLog] = []
    model = None
    best = {"value": float("inf"), "log_hyp": None, "policy": None}
    model_rejections: list[int] = []
    for episode in range(config.episodes):
        model = make_model(scenario, inputs, targets, policy, log_hyp)
        previous_hyp = log_hyp.clone()
        if config.gp_optimizer == "pilco":
            fit_hyperparameters_pilco(model.dynamics,
                                      iterations=config.gp_iterations,
                                      MFEPLS=config.mfephls,
                                      ls_range=config.ls_range)
        else:
            fit_hyperparameters(model.dynamics, iterations=config.gp_iterations,
                                noise_floor=config.noise_floor,
                                ls_range=config.ls_range)
        # --- Step 2: bound how far one episode may reshape the model --------
        if config.hyperparameter_trust_region > 0.0:
            with torch.no_grad():
                delta = (model.dynamics.log_hyp() - previous_hyp).clamp(
                    -config.hyperparameter_trust_region,
                    config.hyperparameter_trust_region)
                model.dynamics.set_log_hyp(previous_hyp + delta)
                model.dynamics.invalidate_cache()
        log_hyp = model.dynamics.log_hyp()

        # --- Step 2c: reject a refit that makes the known-good controller look
        # worse (see PilcoConfig.reject_worse_model) ------------------------
        if config.reject_worse_model and best["log_hyp"] is not None:
            candidate = log_hyp.clone()
            decision = _compare_models(model, plan_state, config, best["log_hyp"])
            if decision is None:
                # a numerical failure while evaluating one of the two models:
                # keep the freshly fitted one and carry on rather than aborting
                model.dynamics.set_log_hyp(candidate)
                log_hyp = candidate
                if log is not None and config.verbose:
                    log("    model comparison failed numerically; keeping the new fit")
            elif decision:
                # Snap back towards the best-so-far fit, then re-fit *briefly* on
                # the current data.  A hard restore would pair hyper-parameters
                # fitted to an earlier, smaller dataset with the current one,
                # which can leave the Gram matrix numerically singular (observed).
                model.dynamics.set_log_hyp(best["log_hyp"])
                try:
                    fit_hyperparameters(model.dynamics, iterations=SNAP_BACK_ITERS,
                                        noise_floor=config.noise_floor,
                                        ls_range=config.ls_range)
                except RuntimeError:
                    model.dynamics.set_log_hyp(best["log_hyp"])
                log_hyp = model.dynamics.log_hyp().clone()
                model_rejections.append(episode)
                if log is not None and config.verbose:
                    log("    model refit rejected: the best-so-far fit gives the "
                        "incumbent controller the lower predicted cost")
            else:
                model.dynamics.set_log_hyp(candidate)
                log_hyp = candidate

        objective = config.objective
        if objective.kind == "gittins":
            objective = ExplorationObjective(
                kind="gittins", beta=objective.beta,
                info_weight=objective.info_weight,
                cross_terms=objective.cross_terms,
                episodes_remaining=max(config.episodes - episode, 1))

        # --- Step 3: baseline = the incumbent controller under the NEW model
        baseline = float("inf")
        if config.accept_if_improved:
            with torch.no_grad():
                baseline = float(plan(model, plan_state, config.horizon).value())
            snapshot = _policy_snapshot(model)

        if log is not None and config.verbose:
            log(f"  episode {episode + 1}/{config.episodes}: "
                f"data={inputs.shape[0]} GP NLML={float(model.dynamics.nlml()):.4e}"
                + (f" baseline={baseline:.3f}" if config.accept_if_improved else ""))

        _, n_iter = optimise_policy(model, plan_state, config, objective,
                                    verbose=config.verbose)
        with torch.no_grad():
            traj = plan(model, plan_state, config.horizon, with_jacobians=True)
            from .objectives import cumulative_cost_variance, information_gain
            pred = float(traj.value())
            var = float(cumulative_cost_variance(traj, model))
            ig = float(information_gain(traj, model))

        # --- Step 3: never keep a controller the model says is worse -------
        accepted = True
        if config.accept_if_improved and pred > baseline:
            accepted = False
            _policy_restore(model, snapshot)
            with torch.no_grad():
                traj = plan(model, plan_state, config.horizon, with_jacobians=True)
                pred = float(traj.value())
                var = float(cumulative_cost_variance(traj, model))
                ig = float(information_gain(traj, model))
            if log is not None and config.verbose:
                log(f"    rejected (predicted {pred:.3f} > baseline "
                    f"{baseline:.3f}); keeping the incumbent controller")

        # --- Step 2b: remember the best model-predicted controller ---------
        # ``best`` is maintained unconditionally: step 2c needs the best-so-far
        # *model* whether or not the caller wants the final rollback.
        if pred < best["value"]:
            best = {"value": pred, "log_hyp": log_hyp.clone(),
                    "policy": _policy_snapshot(model)}

        # --- roll out the true system and log the realised cost -------------
        x0 = (initial_state.mean.numpy()
              + np.linalg.cholesky(initial_state.cov.numpy())
              @ rng.standard_normal(initial_state.dim))
        states, controls, costs = scenario.rollout_true(
            model, x0, config.horizon, rng)
        i, t = scenario.data_from_rollout(states, controls)
        inputs = torch.cat([inputs, torch.as_tensor(i, dtype=DTYPE)])
        targets = torch.cat([targets, torch.as_tensor(t, dtype=DTYPE)])
        logs.append(EpisodeLog(episode, np.asarray(costs), pred,
                               float(np.sqrt(max(var, 0.0))), ig,
                               int(inputs.shape[0]), n_iter,
                               baseline, accepted))
        if log is not None and config.verbose:
            log(f"    realised cost {float(np.sum(costs)):7.2f}   "
                f"predicted {pred:7.2f} +- {np.sqrt(max(var, 0.0)):6.2f}   "
                f"IG {ig:7.2f}" + ("" if accepted else "  [kept incumbent]"))

    if log is not None and config.verbose and model_rejections:
        log(f"  model refits rejected in episodes {model_rejections}")

    # --- Step 2b: hand back the best model-predicted controller ------------
    if config.keep_best_hyperparameters and best["policy"] is not None:
        _policy_restore(model, best["policy"])
        with torch.no_grad():
            model.dynamics.set_log_hyp(best["log_hyp"])

    return model, logs


def _compare_models(model, plan_state, config, other_hyp):
    """Return ``True`` if ``other_hyp`` should replace the current GP fit.

    Compares the two candidate models on the *same* (incumbent) controller.
    Returns ``None`` if either evaluation fails numerically, so that a fragile
    comparison can never abort learning.
    """
    import math

    candidate = model.dynamics.log_hyp().clone()
    try:
        with torch.no_grad():
            j_cand = float(plan(model, plan_state, config.horizon,
                                with_jacobians=False).value())
            model.dynamics.set_log_hyp(other_hyp)
            j_best = float(plan(model, plan_state, config.horizon,
                                with_jacobians=False).value())
    except RuntimeError:
        model.dynamics.set_log_hyp(candidate)
        return None
    model.dynamics.set_log_hyp(candidate)
    if not (math.isfinite(j_cand) and math.isfinite(j_best)):
        return None
    return j_best <= j_cand


def _policy_params(model):
    return [model.policy.centers, model.policy.targets,
            model.policy.log_lengthscales]


def _policy_snapshot(model) -> list:
    return [p.detach().clone() for p in _policy_params(model)]


def _policy_restore(model, snapshot) -> None:
    with torch.no_grad():
        for p, v in zip(_policy_params(model), snapshot):
            p.copy_(v)


def _random_rollout(scenario, x0: np.ndarray, steps: int,
                    rng: np.random.Generator):
    """Initial exploration: uniform random controls (PILCO's ``rollout``)."""

    state = x0.copy()
    states = [state.copy()]
    controls, costs = [], []
    for _ in range(steps):
        u = scenario.max_u * (2.0 * rng.random() - 1.0)
        nxt = scenario.step_true(state, u)
        nxt = nxt + rng.standard_normal(state.shape[0]) * scenario.process_noise
        controls.append(u)
        costs.append(0.0)
        states.append(nxt.copy())
        state = nxt
    return np.asarray(states), np.asarray(controls), np.asarray(costs)
