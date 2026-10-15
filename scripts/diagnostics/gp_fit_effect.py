"""Isolate the closed-loop gap: same policy optimiser, different dynamics fits.

The PyTorch model layer matches the NumPy reference analytically, and on
identical data + identical starting policy the policy optimiser is comparable.
The closed loop nevertheless fails to reach the reference's cart-pole cost.
This script varies *only* the dynamics GP fit and holds everything else fixed:

  A. the hyper-parameters trained by NumPy's ``minimize`` (the reference)
  B. PyTorch L-BFGS, 150 iterations (the current default behaviour)
  C. PyTorch L-BFGS, 30 iterations
  D. PyTorch L-BFGS, 150 iterations, ARD lengthscales **shared across outputs**
     (fewer hyper-parameters: d + 2E instead of d*E + 2E)

For each, the same torch L-BFGS policy optimiser is run from the same start and
the resulting predicted value is reported.  If (B) is much worse than (A), the
gap is the GP fit; if (D) recovers (A), the fix is to reduce the
hyper-parameter count rather than to stop early.

Usage::

    python scripts/diagnostics/gp_fit_effect.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time

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
from pilco_python.util.gaussian import gaussian                           # noqa: E402
from pilco_python.gp.train import train as gp_train                       # noqa: E402
from pilco_python.gp.gp1d import gp1d                                     # noqa: E402

from pilco.gaussian import Gaussian                                       # noqa: E402
from pilco.models.gp import fit_hyperparameters                           # noqa: E402
from pilco.objectives import ExplorationObjective                         # noqa: E402
from pilco.ops.gp import GpDynamicsOp                                     # noqa: E402
from pilco.ops.policy import RbfPolicyOp                                  # noqa: E402
from pilco.pilco import PilcoConfig, optimise_policy                      # noqa: E402
from pilco.scenarios.cartpole import make_cartpole                        # noqa: E402
from pilco.trajectory import PilcoModel, plan                             # noqa: E402

POLICY_ITERS = 60


def build_data(init_steps=40):
    np.random.seed(42)
    s = define_settings()
    plant, cost = s["plant"], s["cost"]
    plant["ctrl"] = "zoh"
    plant["prop"] = propagated
    xx, yy, _, _ = rollout(gaussian(s["mu0"], s["S0"]), s["policy"], init_steps,
                           plant, cost, compute_cost=False)
    pol = {"maxU": s["policy"]["maxU"].copy(), "fcn": ctrl,
           "p": {k: v.copy() for k, v in s["policy"]["p"].items()}}
    dm = {"fcn": gp1d, "train": gp_train, "induce": np.zeros((50, 0, 1))}
    with contextlib.redirect_stdout(io.StringIO()):
        dm = train_dyn_model(xx, yy, dm, pol, plant, [150, 0])
    return s, dm


def make_model(dm, share_lengthscales=False):
    scenario = make_cartpole()
    s_hyp = np.asarray(dm["hyp"])
    d, E = s_hyp.shape[0] - 2, s_hyp.shape[1]
    # a deterministic, non-degenerate starting controller so that every variant
    # is optimised from exactly the same point
    rng = np.random.RandomState(0)
    centers = torch.as_tensor(rng.randn(12, 5) * 0.5)
    policy = RbfPolicyOp(centers, torch.zeros(12, 1),
                         torch.tensor([1.0, 1.0, 1.0, 0.7, 0.7]))
    op = GpDynamicsOp.from_log_hyp(torch.as_tensor(dm["inputs"]),
                                   torch.as_tensor(dm["targets"]),
                                   torch.as_tensor(s_hyp),
                                   share_lengthscales=share_lengthscales)
    model = PilcoModel(scenario.spec, op, policy, scenario.cost,
                       max_u=[scenario.max_u])
    return scenario, model


def evaluate(model, H, budget=POLICY_ITERS):
    x0 = Gaussian(torch.tensor([0.0, 0.0, 0.0, 0.0]),
                  torch.eye(4) * 0.01)
    cfg = PilcoConfig(horizon=H, policy_iterations=budget)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        val, n = optimise_policy(model, x0, cfg, ExplorationObjective(kind="mean"))
    traj = plan(model, x0, H)
    return val, n, time.time() - t0, float(traj.value().detach())


def main() -> int:
    s, dm = build_data()
    H = s["H"]
    print(f"data: {dm['inputs'].shape[0]} points, hyp shape {np.asarray(dm['hyp']).shape}")
    print(f"numpy-trained -NLML = {_np_nlml(dm):.3f}")

    rows = []
    # A: reference fit
    _, model = make_model(dm)
    val, n, dt, _ = evaluate(model, H)
    rows.append(("A numpy minimize hyp", float(model.dynamics.nlml()), val, n, dt))

    # B / C: torch L-BFGS
    for label, iters in (("B torch LBFGS 150", 150), ("C torch LBFGS 30", 30)):
        _, model = make_model(dm)
        fit_hyperparameters(model.dynamics, iterations=iters)
        val, n, dt, _ = evaluate(model, H)
        rows.append((label, float(model.dynamics.nlml()), val, n, dt))

    # D: shared lengthscales (fewer hyper-parameters)
    for label, iters in (("D shared-ell LBFGS 150", 150),
                         ("E shared-ell LBFGS 400", 400)):
        _, model = make_model(dm, share_lengthscales=True)
        fit_hyperparameters(model.dynamics, iterations=iters)
        val, n, dt, _ = evaluate(model, H)
        rows.append((label, float(model.dynamics.nlml()), val, n, dt))

    # F/G/H: noise floor at the plant's process noise (cart-pole uses std 0.01)
    for label, floor in (("F noise floor 0.01", 0.01),
                         ("G noise floor 0.005", 0.005),
                         ("H noise floor 0.02", 0.02)):
        _, model = make_model(dm)
        fit_hyperparameters(model.dynamics, iterations=150, noise_floor=floor)
        val, n, dt, _ = evaluate(model, H)
        rows.append((label, float(model.dynamics.nlml()), val, n, dt))

    # I/J: tighten the lengthscale curb so the GP cannot become arbitrarily
    # smooth (measured: the reference's fit is far smoother than ours)
    for label, rng_ in (("I ls_range 2.0", 2.0), ("J ls_range 3.0", 3.0),
                        ("K ls_range 1.0", 1.0)):
        _, model = make_model(dm)
        fit_hyperparameters(model.dynamics, iterations=150, ls_range=rng_)
        val, n, dt, _ = evaluate(model, H)
        rows.append((label, float(model.dynamics.nlml()), val, n, dt))

    print(f"\n{'variant':26s} {'-NLML':>10s} {'policy J':>10s} {'evals':>6s} {'sec':>6s}")
    print("-" * 64)
    for label, nlml, val, n, dt in rows:
        print(f"{label:26s} {nlml:>10.3f} {val:>10.4f} {n:>6d} {dt:>6.0f}")
    print("\nInterpretation: if A << B in 'policy J' the dynamics fit is the cause;")
    print("if D recovers A, fewer hyper-parameters is the fix rather than early stopping.")
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
        tot += 0.5 * Y[:, e] @ alpha + np.log(np.diag(L)).sum() + 0.5 * len(X) * np.log(2 * np.pi)
    return tot


if __name__ == "__main__":
    raise SystemExit(main())
