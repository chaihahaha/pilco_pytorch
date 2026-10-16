"""Profile the hot paths and measure the effect of the vectorisations.

The closed loop's cost is dominated by objective evaluations, so this breaks a
single ``plan`` and a single objective evaluation into its parts:

1. ``GpDynamicsOp.moments`` — the E x E output loop
2. ``PilcoModel.step_augmented`` — the five splices
3. ``plan`` — H sequential steps, with and without Jacobians
4. the objectives — mean, cumulative-cost variance, information gain
5. the ported ``minimize`` overhead relative to the objective it drives
6. ``torch.compile`` on the one-step map

Usage::

    python scripts/diagnostics/profile_hotspots.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
torch.set_default_dtype(torch.float64)

from pilco.gaussian import Gaussian                                   # noqa: E402
from pilco.objectives import (ExplorationObjective,                   # noqa: E402
                              cumulative_cost_variance, expected_kernel_matrix,
                              information_gain)
from pilco.ops.gp import GpDynamicsOp                                 # noqa: E402
from pilco.scenarios.cartpole import make_cartpole                    # noqa: E402
from pilco.trajectory import PilcoModel, plan                         # noqa: E402


def timeit(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def make_model(n_data=200, seed=0):
    rng = np.random.default_rng(seed)
    scenario = make_cartpole()
    X = rng.standard_normal((n_data, 6)) * 0.5
    Y = rng.standard_normal((n_data, 4)) * 0.05
    hyp = np.zeros((8, 4))
    hyp[:6] = np.log(0.5 + rng.random((6, 4)))
    hyp[6] = np.log(0.5)
    hyp[7] = np.log(0.05)
    dynamics = GpDynamicsOp.from_log_hyp(torch.as_tensor(X), torch.as_tensor(Y),
                                         torch.as_tensor(hyp))
    from pilco.ops.policy import RbfPolicyOp
    centers = torch.as_tensor(rng.standard_normal((12, 5)) * 0.5)
    policy = RbfPolicyOp(centers, torch.zeros(12, 1),
                         torch.tensor([1.0, 1.0, 1.0, 0.7, 0.7]))
    model = PilcoModel(scenario.spec, dynamics, policy, scenario.cost,
                       max_u=[scenario.max_u])
    return scenario, model


def main() -> int:
    scenario, model = make_model()
    H = 40
    x0 = scenario.optimisation_state()
    m, S = x0.mean, x0.cov

    t_step = timeit(lambda: model.step_augmented(m, S))
    t_gp = timeit(lambda: model.dynamics.moments(
        model.dynamics.inputs.mean(0)[:6].detach().clone(),
        torch.eye(6) * 0.1))
    t_plan = timeit(lambda: plan(model, x0, H, with_jacobians=False), n=10)
    t_plan_j = timeit(lambda: plan(model, x0, H, with_jacobians=True), n=5)
    traj = plan(model, x0, H, with_jacobians=True)
    t_mean = timeit(lambda: ExplorationObjective(kind="mean")(traj, model), n=20)
    t_var = timeit(lambda: cumulative_cost_variance(traj, model), n=3)
    t_ig = timeit(lambda: information_gain(traj, model), n=3)

    print(f"one step_augmented        : {t_step*1e3:8.2f} ms")
    print(f"  (of which GP moments)   : {t_gp*1e3:8.2f} ms")
    print(f"plan H={H} (no Jacobians) : {t_plan*1e3:8.2f} ms  "
          f"({t_plan/H*1e3:.2f} ms/step)")
    print(f"plan H={H} (with Jacobians): {t_plan_j*1e3:8.2f} ms  "
          f"(jacfwd overhead x{t_plan_j/t_plan:.2f})")
    print(f"objective mean            : {t_mean*1e6:8.1f} us")
    print(f"objective cost variance   : {t_var*1e3:8.2f} ms  "
          f"({t_var/t_plan*100:.0f}% of a plain plan)")
    print(f"objective information gain: {t_ig*1e3:8.2f} ms  "
          f"({t_ig/t_plan*100:.0f}% of a plain plan)")

    # --- expected_kernel_matrix: how much is the (t,s) double loop? --------
    means, covs = traj.gp_inputs(scenario.spec)
    print(f"\nexpected_kernel_matrix: {means.shape[0]}x{means.shape[0]} pairs, "
          f"6-dim inputs")
    t_ek = timeit(lambda: expected_kernel_matrix(model.dynamics, means, covs, 0),
                  n=3)
    print(f"  per output: {t_ek*1e3:.2f} ms  ->  x4 outputs = {t_ek*4*1e3:.2f} ms")

    # --- torch.compile on the one-step map ---------------------------------
    print("\ntorch.compile(one-step mean map):")
    try:
        compiled = torch.compile(lambda mm: model.step_augmented(mm, S)[0],
                                 dynamic=False)
        compiled(m)
        t_c = timeit(lambda: compiled(m), n=10)
        print(f"  compiled : {t_c*1e3:.2f} ms   eager: {t_step*1e3:.2f} ms   "
              f"speedup x{t_step/t_c:.2f}")
    except Exception as exc:  # pragma: no cover
        print(f"  unavailable: {type(exc).__name__}: {str(exc)[:120]}")

    # --- minimize overhead -------------------------------------------------
    from pilco.optim.minimize import ParamVector, autograd_objective, minimize
    params = [model.policy.centers, model.policy.targets,
              model.policy.log_lengthscales]
    for p in [model.dynamics.log_lengthscales, model.dynamics.log_signal,
              model.dynamics.log_noise]:
        p.requires_grad_(False)
    vec = ParamVector(params)
    evaluate = autograd_objective(
        lambda: plan(model, x0, H, with_jacobians=False).value(), vec)
    t_eval = timeit(lambda: evaluate(vec.flat()), n=5)
    res = minimize(vec.flat(), evaluate, length=5, MFEPLS=8)
    print(f"\nported minimize: one objective evaluation = {t_eval*1e3:.2f} ms; "
          f"length=5/MFEPLS=8 -> {len(res.fX)} evals, "
          f"{len(res.fX)*t_eval*1e3:.0f} ms of pure objective time")
    print("  (the line-search arithmetic itself is small vector ops on ~67 "
          "numbers; the cost is the objective)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
