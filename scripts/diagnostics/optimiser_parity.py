"""Diagnose the policy optimiser: same data, same starting policy, PyTorch vs NumPy.

The forward value and its gradient agree with the NumPy reference to 1e-6
(``tests/test_end_to_end_vs_numpy.py``), so if the optimiser converges to a much
worse objective on identical inputs the difference is the optimiser, not the model.

Usage::

    python scripts/diagnostics/optimiser_parity.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
torch.set_default_dtype(torch.float64)

from pilco_python.scenarios.cartPole.settings_cp import define_settings   # noqa: E402
from pilco_python.scenarios.cartPole.benchmark_de import ctrl             # noqa: E402
from pilco_python.base.rollout import rollout                             # noqa: E402
from pilco_python.base.trainDynModel import train_dyn_model               # noqa: E402
from pilco_python.base.propagated import propagated                       # noqa: E402
from pilco_python.base.value import value as np_value                     # noqa: E402
from pilco_python.util.gaussian import gaussian                           # noqa: E402
from pilco_python.util.minimize import minimize                           # noqa: E402
from pilco_python.gp.train import train as gp_train                       # noqa: E402
from pilco_python.gp.gp1d import gp1d                                     # noqa: E402

from pilco.gaussian import Gaussian                                       # noqa: E402
from pilco.ops.gp import GpDynamicsOp                                     # noqa: E402
from pilco.ops.policy import RbfPolicyOp                                  # noqa: E402
from pilco.pilco import PilcoConfig, optimise_policy                      # noqa: E402
from pilco.objectives import ExplorationObjective                         # noqa: E402
from pilco.scenarios.cartpole import make_cartpole                        # noqa: E402
from pilco.trajectory import PilcoModel, plan                             # noqa: E402


def main() -> int:
    np.random.seed(42)
    s = define_settings()
    plant, cost, H = s["plant"], s["cost"], s["H"]
    plant["ctrl"] = "zoh"
    plant["prop"] = propagated
    mu0Sim = np.zeros(len(s["dyno"]))
    S0Sim = np.diag([0.01] * len(s["dyno"]))

    # --- initial data exactly as settings_cp / benchmark_de does ------------
    xx, yy, _, _ = rollout(gaussian(s["mu0"], s["S0"]), s["policy"], H,
                           plant, cost, compute_cost=False)
    pol = {"maxU": s["policy"]["maxU"].copy(), "fcn": ctrl,
           "p": {k: v.copy() for k, v in s["policy"]["p"].items()}}
    dm = {"fcn": gp1d, "train": gp_train, "induce": np.zeros((50, 0, 1))}
    with contextlib.redirect_stdout(io.StringIO()):
        dm = train_dyn_model(xx, yy, dm, pol, plant, [150, 0])

    # --- NumPy optimiser ---------------------------------------------------
    J0 = float(np_value(pol["p"], mu0Sim, S0Sim, dm, pol, plant, cost, H,
                        compute_gradients=False))
    ref_hist = []
    with contextlib.redirect_stdout(io.StringIO()):
        _, fX, _, _ = minimize(pol["p"], np_value,
                               {"length": 150, "MFEPLS": 30, "verbosity": 0},
                               mu0Sim, S0Sim, dm, pol, plant, cost, H)
    ref_hist = np.asarray(fX).ravel()
    print(f"numpy : J0={J0:.4f}  final={ref_hist[-1]:.4f}  "
          f"evals={ref_hist.size}  hist[:5]={np.round(ref_hist[:5], 2)}")

    # --- PyTorch optimiser on identical inputs -----------------------------
    scenario = make_cartpole()
    centers = torch.as_tensor(s["policy"]["p"]["inputs"])
    targets = torch.as_tensor(s["policy"]["p"]["targets"])
    hyp = np.asarray(s["policy"]["p"]["hyp"]).ravel().copy()
    hyp[-2], hyp[-1] = np.log(1.0), np.log(0.01)
    policy = RbfPolicyOp(centers, targets, torch.as_tensor(np.exp(hyp[:centers.shape[1]])),
                         noise=float(np.exp(hyp[-1])), signal=float(np.exp(hyp[-2])))
    dynamics = GpDynamicsOp.from_log_hyp(torch.as_tensor(dm["inputs"]),
                                         torch.as_tensor(dm["targets"]),
                                         torch.as_tensor(np.asarray(dm["hyp"])))
    model = PilcoModel(scenario.spec, dynamics, policy, scenario.cost,
                       max_u=[scenario.max_u])
    x0 = Gaussian(torch.as_tensor(np.asarray(mu0Sim).ravel()),
                  torch.as_tensor(np.asarray(S0Sim)))
    print(f"torch : J0={float(plan(model, x0, H).value().detach()):.4f}")

    cfg = PilcoConfig(horizon=H, policy_iterations=150)
    with contextlib.redirect_stdout(io.StringIO()):
        val, n = optimise_policy(model, x0, cfg,
                                 ExplorationObjective(kind="mean"))
    print(f"torch : LBFGS -> J={val:.4f} evals={n}")

    # --- gradient-descent sanity check ------------------------------------
    from pilco.trajectory import plan as _plan
    params = [model.policy.centers, model.policy.targets,
              model.policy.log_lengthscales]
    opt = torch.optim.Adam(params, lr=0.02)
    best = float("inf")
    for it in range(400):
        opt.zero_grad()
        Jv = _plan(model, x0, H, with_jacobians=False).value()
        Jv.backward()
        opt.step()
        best = min(best, float(Jv.detach()))
    print(f"torch : Adam(400 it, lr .02) -> best J={best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
