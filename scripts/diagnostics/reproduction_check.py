"""Episode-by-episode reproduction check: pilco_python vs pilco_pytorch.

The operator/optimiser traces (``optimizer_trace.py``) show the maths agrees.
This script checks the *learning loop* the way a debugger would: run both loops
from the same initial data and the same controller, with the RNG routed through
the same stream, and diff the data each episode.

Divergence is reported at three levels so the first one that matters is visible:

1. the **initial dataset** (must be identical: both read the same numpy rollout);
2. the **GP fit** (hyper-parameters and NLML after training on that dataset);
3. the **controller** after one optimisation round, and the trajectory that
   follows from it.

Usage::

    python scripts/diagnostics/reproduction_check.py [--horizon 40] [--bfgs 40]
"""
from __future__ import annotations

import argparse
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
from pilco_python.base.propagated import propagated                       # noqa: E402
from pilco_python.base.trainDynModel import train_dyn_model               # noqa: E402
from pilco_python.base.value import value as np_value                     # noqa: E402
from pilco_python.util.gaussian import gaussian                           # noqa: E402
from pilco_python.util.minimize import minimize                           # noqa: E402
from pilco_python.util.unwrap import unwrap                               # noqa: E402
from pilco_python.gp.train import train as gp_train                       # noqa: E402
from pilco_python.gp.gp1d import gp1d                                     # noqa: E402

from pilco.gaussian import Gaussian                                       # noqa: E402
from pilco.models.gp import (fit_hyperparameters,                         # noqa: E402
                             fit_hyperparameters_pilco)
from pilco.objectives import ExplorationObjective                         # noqa: E402
from pilco.ops.gp import GpDynamicsOp                                     # noqa: E402
from pilco.ops.policy import RbfPolicyOp                                  # noqa: E402
from pilco.pilco import PilcoConfig, optimise_policy                      # noqa: E402
from pilco.scenarios.cartpole import make_cartpole                        # noqa: E402
from pilco.trajectory import PilcoModel, plan                             # noqa: E402


def numpy_initial_data(horizon, seed=42):
    np.random.seed(seed)
    s = define_settings()
    plant, cost = s["plant"], s["cost"]
    plant["ctrl"] = "zoh"
    plant["prop"] = propagated
    with contextlib.redirect_stdout(io.StringIO()):
        xx, yy, _, _ = rollout(gaussian(s["mu0"], s["S0"]), s["policy"], horizon,
                               plant, cost, compute_cost=False)
    return s, plant, cost, xx, yy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--bfgs", type=int, default=40)
    ap.add_argument("--gp-iter", type=int, default=150)
    ap.add_argument("--optimizer", default="pilco", choices=["pilco", "lbfgs"])
    args = ap.parse_args()
    H = args.horizon

    s, plant, cost, xx, yy = numpy_initial_data(H)
    mu0Sim = np.zeros(len(s["dyno"]))
    S0Sim = np.diag([0.01] * len(s["dyno"]))

    # ---------------- 1. initial dataset -------------------------------
    pol = {"maxU": s["policy"]["maxU"].copy(), "fcn": ctrl,
           "p": {k: v.copy() for k, v in s["policy"]["p"].items()}}
    dm = {"fcn": gp1d, "train": gp_train, "induce": np.zeros((50, 0, 1))}
    with contextlib.redirect_stdout(io.StringIO()):
        dm = train_dyn_model(xx, yy, dm, pol, plant, [args.gp_iter, 0])

    scenario = make_cartpole(horizon=H)
    scenario.use_legacy_rng = True
    hyp = np.asarray(pol["p"]["hyp"]).ravel().copy()
    hyp[-2], hyp[-1] = np.log(1.0), np.log(0.01)
    d_poli = pol["p"]["inputs"].shape[1]
    policy = RbfPolicyOp(torch.as_tensor(pol["p"]["inputs"]),
                         torch.as_tensor(pol["p"]["targets"]),
                         torch.as_tensor(np.exp(hyp[:d_poli])),
                         noise=float(np.exp(hyp[-1])),
                         signal=float(np.exp(hyp[-2])))
    dynamics = GpDynamicsOp.from_log_hyp(torch.as_tensor(dm["inputs"]),
                                         torch.as_tensor(dm["targets"]),
                                         torch.as_tensor(np.asarray(dm["hyp"])))
    model = PilcoModel(scenario.spec, dynamics, policy, scenario.cost,
                       max_u=[scenario.max_u])
    x0 = Gaussian(torch.as_tensor(np.asarray(mu0Sim)), torch.as_tensor(np.asarray(S0Sim)))

    j_np = float(np_value(pol["p"], mu0Sim, S0Sim, dm, pol, plant, cost, H,
                          compute_gradients=False))
    with torch.no_grad():
        j_t = float(plan(model, x0, H).value())
    print("1) initial dataset / value")
    print(f"   datasets identical        : "
          f"{np.allclose(dm['inputs'], model.dynamics.inputs.numpy())}")
    print(f"   J numpy={j_np:.10f}  torch={j_t:.10f}  diff={abs(j_np-j_t):.2e}")

    # ---------------- 2. GP fit ----------------------------------------
    print("\n2) GP fit on that dataset (this is where they differ)")
    print(f"   numpy  log ell[out0] = {np.round(np.asarray(dm['hyp'])[:d_poli,0], 4)}")
    print(f"   numpy  -NLML total   = {_np_nlml(dm):.4f}")
    with contextlib.redirect_stdout(io.StringIO()):
        if args.optimizer == "pilco":
            fit_hyperparameters_pilco(model.dynamics, iterations=args.gp_iter,
                                      MFEPLS=30)
        else:
            fit_hyperparameters(model.dynamics, iterations=args.gp_iter)
    print(f"   torch  log ell[out0] = "
          f"{np.round(model.dynamics.log_lengthscales.detach().numpy()[:d_poli,0], 4)}")
    print(f"   torch  -NLML total   = {float(model.dynamics.nlml()):.4f}")
    print(f"   hyper-parameter max abs diff = "
          f"{np.abs(model.dynamics.log_hyp().detach().numpy() - np.asarray(dm['hyp'])).max():.4f}")

    # ---------------- 3. one policy optimisation round ------------------
    print("\n3) one controller optimisation round")
    # numpy: warm-start from the same initial controller
    pol_np = {"maxU": pol["maxU"], "fcn": ctrl,
              "p": {k: v.copy() for k, v in pol["p"].items()}}
    with contextlib.redirect_stdout(io.StringIO()):
        _, fX, _, _ = minimize(pol_np["p"], np_value,
                               {"length": args.bfgs, "MFEPLS": 8, "verbosity": 0},
                               mu0Sim, S0Sim, dm, pol_np, plant, cost, H)
    # torch, but with the *numpy* GP so only the optimiser differs
    model.dynamics.set_log_hyp(torch.as_tensor(np.asarray(dm["hyp"])))
    cfg = PilcoConfig(horizon=H, policy_iterations=args.bfgs,
                      gp_optimizer=args.optimizer,
                      policy_optimizer=args.optimizer)
    with contextlib.redirect_stdout(io.StringIO()):
        j_opt, n = optimise_policy(model, x0, cfg,
                                   ExplorationObjective(kind="mean"))
    print(f"   numpy BFGS  final J = {np.asarray(fX).ravel()[-1]:.6f} "
          f"({len(np.asarray(fX).ravel())} line searches)")
    print(f"   torch LBFGS final J = {j_opt:.6f} ({n} evaluations)")

    # ---------------- 4. the trajectory each controller produces --------
    print("\n4) trajectory from the optimised controllers (same legacy RNG stream)")
    np.random.seed(7)
    x_start = gaussian(s["mu0"], s["S0"])
    xx_np, yy_np, rc_np, _ = rollout(x_start, pol_np, H, plant, cost,
                                     compute_cost=True)
    np.random.seed(7)
    x_start2 = gaussian(s["mu0"], s["S0"])
    st, uu, cc = scenario.rollout_true(model, x_start2, H, None)
    print(f"   per-step cost max abs diff  : {np.abs(np.asarray(rc_np) - cc).max():.3e}")
    print(f"   state max abs diff          : "
          f"{np.abs(xx_np[:H, :4] - st[:H]).max():.3e}")
    print(f"   control max abs diff        : {np.abs(xx_np[:H, -1] - uu).max():.3e}")
    print(f"   cumulative realised cost    : numpy {np.sum(rc_np):.4f}  "
          f"torch {np.sum(cc):.4f}")
    return 0


def _np_nlml(dm):
    from pilco_python.util.maha import maha
    X, Y, hyp = dm["inputs"], dm["targets"], np.asarray(dm["hyp"])
    D = X.shape[1]
    tot = 0.0
    for e in range(Y.shape[1]):
        inp = X / np.exp(hyp[:D, e])
        K = np.exp(2 * hyp[D, e] - maha(inp, inp) / 2)
        A = K + np.exp(2 * hyp[D + 1, e]) * np.eye(len(X))
        L = np.linalg.cholesky(A)
        alpha = np.linalg.solve(A, Y[:, e])
        tot += (0.5 * Y[:, e] @ alpha + np.log(np.diag(L)).sum()
                + 0.5 * len(X) * np.log(2 * np.pi))
    return tot


if __name__ == "__main__":
    raise SystemExit(main())
