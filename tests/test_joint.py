"""Validation of the joint trig augmentation against Monte Carlo.

`ops/joint.py` supplies the cross-covariance blocks used by
`TrigAugmentedCostOp.cross_moment`, and hence by the cumulative-cost-variance
objective.  Every block it computes is a Gaussian integral, so the result should
be *exact* for a jointly Gaussian `(x1, x2)` -- unlike the NumPy reference
`base/analytical_variance.py::augment_cross_covariance`, whose two bugs are
described in `pilco_python/docs/BUGS.md`.

Usage::

    python tests/test_joint.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
torch.set_default_dtype(torch.float64)

from pilco.ops.joint import augmented_joint                # noqa: E402


def main() -> int:
    d1 = d2 = 3
    angles = (0, 2)
    n_mc = 3_000_000
    rng = np.random.default_rng(4)
    m1, m2 = rng.standard_normal(d1), rng.standard_normal(d2)
    a, b = rng.standard_normal((d1, d1)), rng.standard_normal((d2, d2))
    s1 = a @ a.T + 0.3 * np.eye(d1)
    s2 = b @ b.T + 0.3 * np.eye(d2)
    s12 = rng.standard_normal((d1, d2)) * 0.05

    m1t, s1t = torch.as_tensor(m1), torch.as_tensor(s1)
    m2t, s2t = torch.as_tensor(m2), torch.as_tensor(s2)
    s12t = torch.as_tensor(s12)
    _, _, c11, c22, c12 = augmented_joint(m1t, s1t, m2t, s2t, s12t, angles)

    joint = np.block([[s1, s12], [s12.T, s2]]) + 1e-10 * np.eye(d1 + d2)
    L = np.linalg.cholesky(joint)
    z = np.concatenate([m1, m2])[:, None] + L @ rng.standard_normal((d1 + d2, n_mc))
    x1, x2 = z[:d1], z[d1:]

    def aug(x):
        # TrigAugmentOp lays the harmonics out interleaved per angle:
        # [sin(a0), cos(a0), sin(a1), cos(a1), ...]
        parts = [x]
        for a in angles:
            parts.append(np.sin(x[a]))
            parts.append(np.cos(x[a]))
        return np.vstack(parts)

    a1, a2 = aug(x1), aug(x2)
    mc11, mc22 = np.cov(a1), np.cov(a2)
    mc12 = np.cov(np.vstack([a1, a2]))[:a1.shape[0], a1.shape[0]:]

    # The Monte-Carlo standard error of a covariance entry is
    # sigma_i sigma_j / sqrt(n); compare in those units.  A fixed absolute or
    # block-relative tolerance is misleading because the entries span two
    # orders of magnitude (the x-x cross block vs the trig-trig block).
    all_aug = np.vstack([a1, a2])
    diag = np.diag(np.cov(all_aug))
    se = np.sqrt(np.outer(diag[:a1.shape[0]], diag[a1.shape[0]:]) / n_mc)

    ok = True
    for label, analytic, mc in (("marginal x1", c11, mc11),
                                ("marginal x2", c22, mc22)):
        d = np.diag(np.cov(a1 if label.endswith("x1") else a2))
        se_m = np.sqrt(np.outer(d, d) / n_mc)
        z = float((np.abs(analytic.detach().numpy() - mc) / se_m).max())
        good = z < 6.0
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {label:12s} max |err|/se = {z:.2f}")

    z = float((np.abs(c12.detach().numpy() - mc12) / se).max())
    good = z < 6.0
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {'cross block':12s} max |err|/se = {z:.2f}")

    # the state/trig entries for the angle's OWN coordinate: exactly the block
    # the NumPy reference never fills (BUGS.md B3a)
    own = [2 * k for k in range(len(angles))]
    z_own = float((np.abs(c12.detach().numpy()[:d1, d1:][:, own]
                          - mc12[:d1, d1:][:, own]) / se[:d1, d1:][:, own]).max())
    good = z_own < 6.0
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {'own entries':12s} max |err|/se = "
          f"{z_own:.2f}   (the block BUGS.md B3a omits)")

    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
