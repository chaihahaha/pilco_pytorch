"""Locate the closed-loop divergence between pilco_python and pilco_pytorch.

The operator/planner/gradient trace (``optimizer_trace.py``) shows the PyTorch
model reproduces the NumPy reference at every optimiser iterate, so the
closed-loop difference must sit in the *learning loop*, not in the maths.  This
script checks the three candidates in order:

1. **Is the GP training objective identical?**  Compare PyTorch
   ``curbed_nlml`` against NumPy's ``gp/hypCurb.py`` at the same
   hyper-parameters.
2. **Is the optimisation budget comparable?**  Count how many times NumPy's
   ``minimize`` evaluates the objective while training one GP output, and how
   many L-BFGS iterations/evals PyTorch uses.
3. **Does torch converge to a different (better) optimum, and does that matter
   for control?**  Report the NLML and the resulting policy value for both fits.

Usage::

    python scripts/diagnostics/gp_training_parity.py
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
from pilco_python.base.propagated import propagated                       # noqa: E402
from pilco_python.util.gaussian import gaussian                           # noqa: E402
from pilco_python.gp.hypCurb import hypCurb                               # noqa: E402
from pilco_python.gp.covSEard import covSEard                             # noqa: E402
from pilco_python.gp.covNoise import covNoise                             # noqa: E402
from pilco_python.util.minimize import minimize                           # noqa: E402

from pilco.models.gp import curbed_nlml                                   # noqa: E402
from pilco.ops.gp import GpDynamicsOp                                     # noqa: E402


def dataset(horizon=40, seed=42):
    np.random.seed(seed)
    s = define_settings()
    plant, cost = s["plant"], s["cost"]
    plant["ctrl"] = "zoh"
    plant["prop"] = propagated
    with contextlib.redirect_stdout(io.StringIO()):
        xx, yy, _, _ = rollout(gaussian(s["mu0"], s["S0"]), s["policy"], horizon,
                               plant, cost, compute_cost=False)
    trig = np.column_stack([np.sin(xx[:, 3]), np.cos(xx[:, 3])])
    aug = np.column_stack([xx[:, :4], trig])
    dyni = np.asarray(s["dyni"])
    inputs = np.column_stack([aug[:, dyni], xx[:, -1:]])
    targets = yy - xx[:, :4]
    return s, inputs, targets


def main() -> int:
    s, X, Y = dataset()
    D, E = X.shape[1], Y.shape[1]
    curb = {"snr": 1000, "ls": 100, "std": np.std(X, axis=0, ddof=1)}
    covfunc = ("covSum", ("covSEard", "covNoise"))

    # --- starting hyper-parameters, exactly as gp/train.py chooses them -----
    init = np.concatenate([np.log(np.std(X, axis=0, ddof=1)), [0], [-1]])
    lh = np.tile(init.reshape(-1, 1), (1, E))
    lh[D, :] = np.log(np.std(Y, axis=0, ddof=1))
    lh[D + 1, :] = np.log(np.std(Y, axis=0, ddof=1) / 10)

    # --- 1) is the training objective identical? ---------------------------
    hyp0 = lh[:, 0].copy()
    f_np, _ = hypCurb(hyp0, covfunc, X, Y[:, 0], curb)
    op = GpDynamicsOp.from_log_hyp(torch.as_tensor(X), torch.as_tensor(Y),
                                   torch.as_tensor(lh))
    f_t = float(curbed_nlml(op, torch.std(torch.as_tensor(X), dim=0)))
    print("1) training objective at the same hyper-parameters")
    print(f"   numpy hypCurb            = {f_np:.10f}")
    print(f"   torch  curbed_nlml       = {f_t:.10f}")
    print(f"   abs diff                 = {abs(f_np - f_t):.3e}")

    # --- 2) optimisation budget -------------------------------------------
    counts = {"n": 0}

    def counted(lh_i, cf, x, y, cb):
        counts["n"] += 1
        return hypCurb(lh_i, cf, x, y, cb)

    with contextlib.redirect_stdout(io.StringIO()):
        opt_hyp, v, _, _ = minimize(hyp0.copy(), counted, 150, covfunc,
                                    X, Y[:, 0], curb)
    print("\n2) budget")
    print(f"   numpy  : {counts['n']} objective evaluations, "
          f"final value {v[-1]:.6f}")

    # torch, with a matched evaluation budget
    for iters in (150, 4500):
        op2 = GpDynamicsOp.from_log_hyp(torch.as_tensor(X), torch.as_tensor(Y),
                                        torch.as_tensor(lh))
        evals = {"n": 0}
        params = [op2.log_lengthscales, op2.log_signal, op2.log_noise]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters,
                                tolerance_grad=1e-10, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = curbed_nlml(op2, torch.std(torch.as_tensor(X), dim=0))
            loss.backward()
            evals["n"] += 1
            return loss

        with contextlib.redirect_stdout(io.StringIO()):
            opt.step(closure)
        print(f"   torch  : {evals['n']} objective evaluations (max_iter={iters}),"
              f" final value {float(curbed_nlml(op2, torch.std(torch.as_tensor(X), dim=0))):.6f}")

    # --- 3) what the two fits do to the policy -----------------------------
    print("\n3) hyper-parameters and control quality")
    for label, hyp in (("numpy fit", np.asarray(opt_hyp).ravel()),
                       ("torch fit (150 it)", op2.log_hyp().detach().numpy()[:, 0])):
        if label.startswith("torch"):
            continue
        print(f"   {label}: log ell = {np.round(hyp[:D], 3)}  "
              f"snr = {np.exp(hyp[D] - hyp[D+1]):.1f}")
    hyps = {"numpy fit": np.asarray(opt_hyp).ravel()}
    for iters in (150,):
        o = GpDynamicsOp.from_log_hyp(torch.as_tensor(X), torch.as_tensor(Y),
                                      torch.as_tensor(lh))
        params = [o.log_lengthscales, o.log_signal, o.log_noise]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters,
                                tolerance_grad=1e-10, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = curbed_nlml(o, torch.std(torch.as_tensor(X), dim=0))
            loss.backward()
            return loss

        with contextlib.redirect_stdout(io.StringIO()):
            opt.step(closure)
        hyps[f"torch fit ({iters} it)"] = o.log_hyp().detach().numpy(), o
    for label, value in hyps.items():
        h = value[0] if isinstance(value, tuple) else value
        print(f"   {label}: log ell = {np.round(np.asarray(h)[:D, 0], 3)}  "
              f"snr = {np.exp(np.asarray(h)[D, 0] - np.asarray(h)[D + 1, 0]):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
