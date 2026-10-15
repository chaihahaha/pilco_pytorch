"""Step-by-step comparison of PILCO's BFGS and torch L-BFGS on identical inputs.

This answers two questions directly.

1. **Does the PyTorch rewrite reproduce the NumPy reference along the whole
   optimiser trajectory, not just at one point?**  Every NumPy line-search
   iterate ``psi_k`` is re-evaluated in torch and the objective *and* the full
   gradient are compared.  If they agree at every iterate, autograd does not
   accumulate drift and the closed-loop difference must come from elsewhere.
2. **Where do the two optimisers diverge?**  Both sequences are recorded
   (objective, gradient norm) so the first divergence is visible.

The NumPy reference counts **line searches** (``p['length']``), each allowed up
to ``MFEPLS`` function evaluations; torch's ``LBFGS(max_iter=N)`` counts
iterations with a strong-Wolfe line search.  These are not the same budget.

Usage::

    python scripts/diagnostics/optimizer_trace.py [--horizon 20] [--iters 40]
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
from pilco_python.base.trainDynModel import train_dyn_model               # noqa: E402
from pilco_python.base.propagated import propagated                       # noqa: E402
from pilco_python.base.value import value as np_value                     # noqa: E402
from pilco_python.util.gaussian import gaussian                           # noqa: E402
from pilco_python.util.minimize import minimize                           # noqa: E402
from pilco_python.util.unwrap import unwrap                               # noqa: E402
from pilco_python.gp.train import train as gp_train                       # noqa: E402
from pilco_python.gp.gp1d import gp1d                                     # noqa: E402

from pilco.gaussian import Gaussian                                       # noqa: E402
from pilco.ops.gp import GpDynamicsOp                                     # noqa: E402
from pilco.ops.policy import RbfPolicyOp                                  # noqa: E402
from pilco.scenarios.cartpole import make_cartpole                        # noqa: E402
from pilco.trajectory import PilcoModel, plan                             # noqa: E402

#: numpy's parameter layout is [hyp(7), inputs, targets]; only the first
#: ``d_poli`` entries of hyp are free (the other two are signal/noise, which
#: ``congp`` fixes and whose derivatives it zeroes).
N_HYP = 7


def setup(horizon):
    np.random.seed(42)
    s = define_settings()
    plant, cost = s["plant"], s["cost"]
    plant["ctrl"] = "zoh"
    plant["prop"] = propagated
    mu0Sim = np.zeros(len(s["dyno"]))
    S0Sim = np.diag([0.01] * len(s["dyno"]))

    with contextlib.redirect_stdout(io.StringIO()):
        xx, yy, _, _ = rollout(gaussian(s["mu0"], s["S0"]), s["policy"], horizon,
                               plant, cost, compute_cost=False)
    pol = {"maxU": s["policy"]["maxU"].copy(), "fcn": ctrl,
           "p": {k: v.copy() for k, v in s["policy"]["p"].items()}}
    dm = {"fcn": gp1d, "train": gp_train, "induce": np.zeros((50, 0, 1))}
    with contextlib.redirect_stdout(io.StringIO()):
        dm = train_dyn_model(xx, yy, dm, pol, plant, [80, 0])

    scenario = make_cartpole()
    hyp = np.asarray(pol["p"]["hyp"]).ravel().copy()
    hyp[-2], hyp[-1] = np.log(1.0), np.log(0.01)     # congp fixes these
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
    x0 = Gaussian(torch.as_tensor(np.asarray(mu0Sim).ravel()),
                  torch.as_tensor(np.asarray(S0Sim)))
    return horizon, mu0Sim, S0Sim, pol, dm, plant, cost, model, x0


def make_torch_eval(model, x0, H):
    """Return ``psi_numpy -> (J, grad)`` with numpy's parameter layout."""
    d_poli = model.policy.n_in
    nc = model.policy.centers.shape[0]
    n_out = model.policy.n_out
    n_c, n_t = nc * d_poli, nc * n_out

    def unpack(psi):
        hyp = psi[:N_HYP]
        # numpy flattens arrays in Fortran order, so the (nc, d) matrix is the
        # C-order reshape of the *transpose*
        inp = psi[N_HYP:N_HYP + n_c].reshape(d_poli, nc).transpose(1, 0)
        tgt = psi[N_HYP + n_c:N_HYP + n_c + n_t].reshape(n_out, nc).transpose(1, 0)
        return hyp, inp, tgt

    def evaluate(psi):
        hyp, inp, tgt = unpack(np.asarray(psi))
        with torch.no_grad():
            model.policy.log_lengthscales.copy_(torch.as_tensor(hyp[:d_poli]))
            model.policy.centers.copy_(torch.as_tensor(inp))
            model.policy.targets.copy_(torch.as_tensor(tgt))
        J = plan(model, x0, H).value()
        params = [model.policy.centers, model.policy.targets,
                  model.policy.log_lengthscales]
        grads = torch.autograd.grad(J, params)
        # numpy's hyp has 7 entries; the last two (signal std, noise std) are
        # pinned by congp, which also zeroes their derivatives
        zeros = torch.zeros(N_HYP - d_poli, dtype=torch.float64)
        g = torch.cat([grads[2],                                # lengthscales
                       zeros,
                       grads[0].transpose(0, 1).reshape(-1),   # Fortran order
                       grads[1].transpose(0, 1).reshape(-1)])
        return float(J.detach()), g.detach().numpy()

    return evaluate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()
    H = args.horizon

    H, mu0Sim, S0Sim, pol, dm, plant, cost, model, x0 = setup(H)
    torch_eval = make_torch_eval(model, x0, H)

    records = []

    def wrapped(p, *a, **kw):
        J, dJ = np_value(p, *a, compute_gradients=True, **kw)
        psi = unwrap(p).copy()
        g = unwrap(dJ).copy()
        rec = {"psi": psi.copy(), "J": float(J),
               "g_norm": float(np.linalg.norm(g))}
        if len(records) % max(args.every, 1) == 0:
            tJ, tg = torch_eval(psi)
            rec["torch_J"] = tJ
            rec["dJ"] = abs(tJ - float(J))
            rec["dg_rel"] = float(np.linalg.norm(tg - g)
                                  / max(np.linalg.norm(g), 1e-12))
        records.append(rec)
        return J, dJ

    with contextlib.redirect_stdout(io.StringIO()):
        minimize(pol["p"], wrapped,
                 {"length": args.iters, "MFEPLS": 30, "verbosity": 0},
                 mu0Sim, S0Sim, dm, pol, plant, cost, H)

    print(f"numpy BFGS (length={args.iters} line searches): {len(records)} "
          f"objective evaluations, J {records[0]['J']:.6f} -> "
          f"{records[-1]['J']:.6f}")
    checked = [r for r in records if "dJ" in r]
    if checked:
        print(f"torch re-evaluation at the SAME {len(checked)} iterates:")
        print(f"  max |J_torch - J_numpy|     = "
              f"{max(r['dJ'] for r in checked):.3e}")
        print(f"  max rel |g_torch - g_numpy| = "
              f"{max(r['dg_rel'] for r in checked):.3e}")

    print("\n numpy trace (every %d-th):" % max(len(records) // 12, 1))
    prev = None
    for i, r in enumerate(records):
        if i % max(len(records) // 12, 1) == 0 or i == len(records) - 1:
            st = "-" if prev is None else f"{np.linalg.norm(r['psi'] - prev):.3e}"
            extra = (f"   |dJ|={r['dJ']:.1e} |dg|rel={r['dg_rel']:.1e}"
                     if "dJ" in r else "")
            print(f"  {i:4d}  J={r['J']:9.6f}  |g|={r['g_norm']:8.4f}  "
                  f"|step|={st:>9s}{extra}")
        prev = r["psi"]

    # ---- torch L-BFGS sequence ------------------------------------------
    trec = []
    params = [model.policy.centers, model.policy.targets,
              model.policy.log_lengthscales]
    gp_params = [model.dynamics.log_lengthscales, model.dynamics.log_signal,
                 model.dynamics.log_noise]
    for p in gp_params:
        p.requires_grad_(False)
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=args.iters,
                            tolerance_grad=1e-7, tolerance_change=1e-11,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        J = plan(model, x0, H, with_jacobians=False).value()
        J.backward()
        g = float(sum(float(q.grad.norm()) ** 2 for q in params) ** 0.5)
        trec.append({"J": float(J.detach()), "g_norm": g})
        return J

    opt.step(closure)
    for p in gp_params:
        p.requires_grad_(True)
    print(f"\ntorch LBFGS: {len(trec)} closure evaluations, "
          f"J {trec[0]['J']:.6f} -> {trec[-1]['J']:.6f}")

    n = min(len(records), len(trec))
    print(f"\n side-by-side objective sequence (first {n}):")
    print(f"  {'k':>4} {'numpy J':>14} {'torch J':>14} {'numpy-torch':>12}")
    for k in range(0, n, max(n // 10, 1)):
        print(f"  {k:>4} {records[k]['J']:>14.6f} {trec[k]['J']:>14.6f} "
              f"{records[k]['J'] - trec[k]['J']:>12.2e}")
    print(f"\n numpy final J = {records[-1]['J']:.6f}  "
          f"({len(records)} objective evaluations)")
    print(f" torch final J = {trec[-1]['J']:.6f}  ({len(trec)} evaluations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
