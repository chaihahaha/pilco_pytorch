"""Validate the PyTorch moment-matching operators against the NumPy reference.

The references live in ``pilco_python`` (the original port):

* ``util/gTrig.py``  -> :class:`pilco.ops.fourier.TrigAugmentOp`
* ``util/gSat.py``   -> :class:`pilco.ops.fourier.SaturationOp`
* ``gp/gp0d.py``     -> :class:`pilco.ops.gp.GpDynamicsOp`
* ``loss/lossSat.py``-> :class:`pilco.ops.cost.SaturatingCostOp`
* ``control/conlin.py`` -> (affine) and the RBF controller

If the PyTorch operators reproduce those to machine precision, the closed-form
algebra was transcribed correctly, and autograd then automatically supplies the
derivatives that the original code hand-codes.

Usage::

    python tests/test_ops_vs_numpy.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REF = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_ROOT, _REF):
    if p not in sys.path:
        sys.path.insert(0, p)

import types  # noqa: E402

from pilco_python.util.gTrig import gTrig                      # noqa: E402
from pilco_python.util.gSat import gSat                        # noqa: E402
from pilco_python.gp.gp0d import gp0d                          # noqa: E402
from pilco_python.loss.lossSat import lossSat                  # noqa: E402

from pilco.ops.cost import SaturatingCostOp                    # noqa: E402
from pilco.ops.fourier import SaturationOp, TrigAugmentOp      # noqa: E402
from pilco.ops.gp import GpDynamicsOp                          # noqa: E402

torch.set_default_dtype(torch.float64)
TOL = 1e-9


def _random_gaussian(d, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(d)
    a = rng.standard_normal((d, d))
    s = a @ a.T + 0.25 * np.eye(d)
    return m, s


def check_trig(d=4, angles=(1, 3), seed=0):
    m, s = _random_gaussian(d, seed)
    M_ref, V_ref, C_ref = gTrig(m, s, np.array(angles), compute_derivatives=False)
    out = TrigAugmentOp(angles, n_in=d)(torch.as_tensor(m), torch.as_tensor(s))
    return {"mean": _e(out.mean, M_ref), "cov": _e(out.cov, V_ref),
            "cross": _e(out.cross, C_ref)}


def check_sat(d=4, channels=(0, 1, 3), max_u=(1.0, 2.0, 0.5), seed=1):
    m, s = _random_gaussian(d, seed)
    mu = np.asarray(max_u)
    ch = list(channels)
    M_ref, V_ref, C_ref = gSat(m, s, np.array(channels), mu, compute_derivatives=False)
    op = SaturationOp(n_in=len(ch), channel_indices=list(range(len(ch))), max_u=mu)
    out = op(torch.as_tensor(m[ch]),
             torch.as_tensor(s[np.ix_(ch, ch)]))
    # the operator only sees the channel sub-block, so its C is the matching
    # sub-block of gSat's C
    return {"mean": _e(out.mean, M_ref), "cov": _e(out.cov, V_ref),
            "cross": _e(out.cross, C_ref[ch, :])}


def check_gp(d=5, E=4, n=25, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, E))
    hyp = np.zeros((d + 2, E))
    hyp[:d] = np.log(0.5 + rng.random((d, E)))
    hyp[d] = np.log(1.0)
    hyp[d + 1] = np.log(0.1)
    m, s = _random_gaussian(d, seed + 1)

    ref_model = types.SimpleNamespace(hyp=hyp, inputs=X, targets=Y)
    M_ref, S_ref, V_ref = gp0d(ref_model, m, s, compute_derivatives=False)
    op = GpDynamicsOp.from_log_hyp(torch.as_tensor(X), torch.as_tensor(Y),
                                   torch.as_tensor(hyp))
    out = op(torch.as_tensor(m), torch.as_tensor(s))
    return {"mean": _e(out.mean, M_ref), "cov": _e(out.cov, S_ref),
            "cross": _e(out.cross, V_ref)}


def check_cost(d=4, seed=3):
    m, s = _random_gaussian(d, seed)
    rng = np.random.default_rng(seed + 5)
    z = rng.standard_normal(d)
    W = np.diag(0.3 + rng.random(d))
    L_ref, _, _, S2_ref, _, _, _, _, _ = lossSat({"z": z, "W": W}, m, s,
                                                 compute_derivatives=True)
    op = SaturatingCostOp(torch.as_tensor(z), torch.as_tensor(W))
    val, var = op.moments(torch.as_tensor(m), torch.as_tensor(s))
    return {"value": _e(val, np.asarray(L_ref).reshape(-1)), "var": _e(var, np.asarray(S2_ref).reshape(-1))}


def check_cost_cross(d=4, seed=11, n_mc=400_000):
    """Cross moment against an independent Monte-Carlo reference.

    Deliberately *not* compared against a second copy of the same algebra: the
    first version of this test encoded the same sign error as the
    implementation and "passed".  Sampling the joint Gaussian and evaluating the
    cost directly is the only reference that cannot share a bug.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(d)
    W = np.diag(0.3 + rng.random(d))
    m1, s1 = _random_gaussian(d, seed + 1)
    m2, s2 = _random_gaussian(d, seed + 2)
    a = rng.standard_normal((d, d))
    s12 = 0.15 * (a @ a.T)

    joint_cov = np.block([[s1, s12], [s12.T, s2]])
    joint_cov = joint_cov + 1e-8 * np.eye(2 * d)
    L = np.linalg.cholesky(joint_cov)
    samples = np.concatenate([m1, m2])[:, None] + L @ rng.standard_normal((2 * d, n_mc))
    x1, x2 = samples[:d], samples[d:]
    d1, d2 = x1 - z[:, None], x2 - z[:, None]
    q1 = np.einsum("dn,de,en->n", d1, W, d1)
    q2 = np.einsum("dn,de,en->n", d2, W, d2)
    c1, c2 = 1.0 - np.exp(-0.5 * q1), 1.0 - np.exp(-0.5 * q2)
    ref = float(np.mean(c1 * c2))

    op = SaturatingCostOp(torch.as_tensor(z), torch.as_tensor(W))
    got = float(op.cross_moment(torch.as_tensor(m1), torch.as_tensor(m2),
                                torch.as_tensor(s1), torch.as_tensor(s2),
                                torch.as_tensor(s12)))
    return {"cross_moment": abs(got - ref)}


def check_autograd(d=4, angles=(1, 3), seed=2, eps=1e-6):
    """autograd vs central differences through the trig operator."""
    m, s = _random_gaussian(d, seed)
    mt = torch.tensor(m, requires_grad=True)
    st = torch.tensor(s, requires_grad=True)
    op = TrigAugmentOp(angles, n_in=d)
    out = op(mt, st)
    loss = out.mean.sum() + out.cov.sum() + out.cross.sum()
    gm, gs = torch.autograd.grad(loss, (mt, st))

    def f(mv, sv):
        o = op(torch.as_tensor(mv), torch.as_tensor(sv))
        return float(o.mean.sum() + o.cov.sum() + o.cross.sum())

    fd_m = np.zeros(d)
    for i in range(d):
        mp, mm = m.copy(), m.copy(); mp[i] += eps; mm[i] -= eps
        fd_m[i] = (f(mp, s) - f(mm, s)) / (2 * eps)
    fd_s = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            sp, sm = s.copy(), s.copy()
            sp[i, j] += eps; sp[j, i] += eps
            sm[i, j] -= eps; sm[j, i] -= eps
            fd_s[i, j] = (f(m, sp) - f(m, sm)) / (2 * eps)
    gs_sym = gs.numpy() + gs.numpy().T   # S treated as a symmetric variable
    return {"grad_mean": float(np.abs(gm.numpy() - fd_m).max()),
            "grad_cov": float(np.abs(gs_sym - fd_s).max())}


def _e(got, ref):
    return float(np.abs(np.asarray(got.detach()) - np.asarray(ref)).max())


#: (label, callable, tolerance).  The tolerances reflect the reference: exact
#: analytic identities are compared to machine precision, Monte-Carlo
#: references only to sampling error.
CHECKS = [
    ("trig (gTrig)", check_trig, 1e-9),
    ("sat  (gSat)", check_sat, 1e-9),
    ("gp   (gp0d)", check_gp, 1e-8),
    ("cost (lossSat)", check_cost, 1e-9),
    ("cost cross-moment (MC)", check_cost_cross, 5e-3),
    ("autograd vs FD", check_autograd, 1e-6),
]


def main() -> int:
    ok = True
    for name, fn, tol in CHECKS:
        errs = fn()
        worst = max(errs.values())
        good = worst < tol
        ok &= good
        detail = "  ".join(f"{k}={v:.2e}" for k, v in errs.items())
        print(f"[{'PASS' if good else 'FAIL'}] {name:24s} {detail}")
    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
