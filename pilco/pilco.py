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
from .models.gp import build_dynamics_op, fit_hyperparameters, initial_log_hyp
from .objectives import ExplorationObjective
from .ops.policy import RbfPolicyOp
from .trajectory import PilcoModel, plan

__all__ = ["PilcoConfig", "initial_policy", "optimise_policy", "learn",
           "EpisodeLog"]

DTYPE = torch.float64


@dataclass
class PilcoConfig:
    """Hyper-parameters of the learning loop."""

    horizon: int = 40
    episodes: int = 6
    initial_rollouts: int = 1
    gp_iterations: int = 200
    policy_iterations: int = 150
    objective: ExplorationObjective = field(default_factory=ExplorationObjective)
    seed: int = 0
    verbose: bool = True
    #: re-fit hyper-parameters from scratch every episode (PILCO warm-starts)
    retrain_from_scratch: bool = False


@dataclass
class EpisodeLog:
    episode: int
    realised_costs: np.ndarray
    predicted_value: float
    predicted_sd: float
    information_gain: float
    n_data: int
    policy_iterations: int


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
    """L-BFGS on the controller parameters.  Returns the objective value."""
    params = [model.policy.centers, model.policy.targets,
              model.policy.log_lengthscales]
    gp_params = [model.dynamics.log_lengthscales, model.dynamics.log_signal,
                 model.dynamics.log_noise]
    for p in gp_params:
        p.requires_grad_(False)

    optimiser = torch.optim.LBFGS(params, lr=1.0,
                                  max_iter=config.policy_iterations,
                                  tolerance_grad=1e-7, tolerance_change=1e-11,
                                  line_search_fn="strong_wolfe")
    state = {"value": float("inf"), "n": 0}

    def closure():
        optimiser.zero_grad()
        traj = plan(model, initial_state, config.horizon, with_jacobians=True)
        score = objective(traj, model)
        score.backward()
        # autograd can leave a non-contiguous .grad (e.g. from a transpose in
        # the moment formulas), which torch.optim.LBFGS rejects; normalise it
        for p in params:
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()
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
    for episode in range(config.episodes):
        model = make_model(scenario, inputs, targets, policy, log_hyp)
        fit_hyperparameters(model.dynamics, iterations=config.gp_iterations)
        log_hyp = model.dynamics.log_hyp()

        objective = config.objective
        if objective.kind == "gittins":
            objective = ExplorationObjective(
                kind="gittins", beta=objective.beta,
                info_weight=objective.info_weight,
                cross_terms=objective.cross_terms,
                episodes_remaining=max(config.episodes - episode, 1))
        if log is not None and config.verbose:
            log(f"  episode {episode + 1}/{config.episodes}: "
                f"data={inputs.shape[0]} GP NLML={float(model.dynamics.nlml()):.4e}")

        value, n_iter = optimise_policy(model, plan_state, config, objective,
                                        verbose=config.verbose)
        # report the *diagnostic* quantities for the chosen controller
        with torch.no_grad():
            traj = plan(model, plan_state, config.horizon, with_jacobians=True)
            from .objectives import cumulative_cost_variance, information_gain
            pred = float(traj.value())
            var = float(cumulative_cost_variance(traj, model))
            ig = float(information_gain(traj, model))

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
                               int(inputs.shape[0]), n_iter))
        if log is not None and config.verbose:
            log(f"    realised cost {float(np.sum(costs)):7.2f}   "
                f"predicted {pred:7.2f} +- {np.sqrt(max(var, 0.0)):6.2f}   "
                f"IG {ig:7.2f}")

    return model, logs


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
