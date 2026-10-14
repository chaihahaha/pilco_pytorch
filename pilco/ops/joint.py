"""Joint trigaugmentation of two correlated Gaussian states.

PILCO evaluates the cost on the *trig-augmented* state ``[x | sin(theta),
cos(theta)]``.  To get the closed-form cumulative-cost variance we need the
joint law of the augmented states at two different timesteps, i.e. the
covariance of ``[x_t, sin(t), cos(t), x_t', sin(t'), cos(t')]``.

The marginal blocks come from :class:`~pilco.ops.fourier.TrigAugmentOp`; the
only new work is the cross blocks, all of which are Gaussian integrals:

* ``Cov[x_b, sin(x2_a)] = Cov[x1_b, x2_a] e^{-S22[a,a]/2} cos(m2_a)``
* ``Cov[x_b, cos(x2_a)] = -Cov[x1_b, x2_a] e^{-S22[a,a]/2} sin(m2_a)``
* for two angles ``(a1 in x1, a2 in x2)`` with ``mu=m1[a1]``, ``nu=m2[a2]``,
  ``p=S11[a1,a1]``, ``q=S22[a2,a2]``, ``r=S12[a1,a2]`` and
  ``D = e^{-(p+q-2r)/2}``, ``P = e^{-(p+q+2r)/2}``::

      E[sin mu sin nu] = 1/2 [ cos(mu-nu) D - cos(mu+nu) P ]
      E[cos mu cos nu] = 1/2 [ cos(mu-nu) D + cos(mu+nu) P ]
      E[sin mu cos nu] = 1/2 [ sin(mu+nu) P + sin(mu-nu) D ]
      E[cos mu sin nu] = 1/2 [ sin(mu+nu) P - sin(mu-nu) D ]

``base/analytical_variance.py::augment_cross_covariance`` implements the same
thing, but the audit found two bugs there (it skips the angle's own coordinate
when filling the state/trig cross block, and it assumes ``S12`` is symmetric
when transposing).  The version below is written directly from the formulas and
is validated against a Monte-Carlo reference in the test suite.
"""
from __future__ import annotations

from typing import Sequence

import torch

from ..gaussian import symmetrize

__all__ = ["augmented_joint", "other_trig_cross"]


def other_trig_cross(mean: torch.Tensor, cov: torch.Tensor,
                     cross_cov: torch.Tensor, angles: Sequence[int]):
    """Blocks ``Cov[x_other, trig(x)]`` and ``Cov[trig(x), x_other]``.

    Parameters
    ----------
    mean, cov:
        The Gaussian ``x`` whose trig augmentation is wanted.
    cross_cov:
        ``Cov[x, x_other]``, shape ``(len(x), n_other)``.
    angles:
        Angle coordinates of ``x``.

    Returns
    -------
    ``(Cov[x_other, trig(x)], Cov[trig(x), x_other])`` with shapes
    ``(n_other, 2A)`` and ``(2A, n_other)``.
    """
    angles = list(angles)
    A = len(angles)
    decay = torch.exp(-0.5 * torch.diagonal(cov)[angles])
    sin_m, cos_m = torch.sin(mean[angles]), torch.cos(mean[angles])
    left = cross_cov[angles].transpose(0, 1)             # (n_other, A)
    # TrigAugmentOp interleaves the harmonics: [sin(a0), cos(a0), sin(a1), ...]
    out = torch.empty(left.shape[0], 2 * A, dtype=left.dtype, device=left.device)
    out[:, 0::2] = left * (decay * cos_m)[None, :]       # Cov[., sin]
    out[:, 1::2] = left * (-decay * sin_m)[None, :]      # Cov[., cos]
    return out, out[:, 0::2].transpose(0, 1)


def augmented_joint(mean1: torch.Tensor, cov1: torch.Tensor,
                    mean2: torch.Tensor, cov2: torch.Tensor,
                    cross: torch.Tensor, angles: Sequence[int]):
    """Joint law of ``[x1, trig(x1)]`` and ``[x2, trig(x2)]``.

    ``cross`` is ``Cov[x1, x2]``.  Returns
    ``(aug_mean1, aug_mean2, aug_cov11, aug_cov22, aug_cov12)`` where
    ``aug_cov12 = Cov[[x1, trig(x1)], [x2, trig(x2)]]``.
    """
    from .fourier import TrigAugmentOp

    d = mean1.numel()
    A = len(angles)
    op = TrigAugmentOp(angles, n_in=d)
    t1 = op.moments(mean1, cov1)
    t2 = op.moments(mean2, cov2)
    aug_mean1 = torch.cat([mean1, t1.mean])
    aug_mean2 = torch.cat([mean2, t2.mean])

    def _marg(cov, trig):
        proj = cov @ trig.cross                       # Cov[state, trig]
        out = torch.zeros(d + 2 * A, d + 2 * A, dtype=cov.dtype, device=cov.device)
        out[:d, :d] = cov
        out[:d, d:] = proj
        out[d:, :d] = proj.transpose(0, 1)
        out[d:, d:] = trig.cov
        return out

    aug_cov11 = _marg(cov1, t1)
    aug_cov22 = _marg(cov2, t2)

    aug_cov12 = torch.zeros(d + 2 * A, d + 2 * A,
                            dtype=cov1.dtype, device=cov1.device)
    aug_cov12[:d, :d] = cross
    # Cov[x1, trig(x2)]: x = x2, x_other = x1, so the cross-covariance is
    # Cov[x2, x1] = cross^T
    c_x1_trig2, _ = other_trig_cross(mean2, cov2, cross.transpose(0, 1), angles)
    aug_cov12[:d, d:] = c_x1_trig2
    # Cov[trig(x1), x2]: x = x1, x_other = x2, cross-covariance is cross
    c_x2_trig1, _ = other_trig_cross(mean1, cov1, cross, angles)
    aug_cov12[d:, :d] = c_x2_trig1.transpose(0, 1)
    aug_cov12[d:, d:] = _trig_trig_cross(mean1, cov1, mean2, cov2, cross, angles)
    return aug_mean1, aug_mean2, aug_cov11, aug_cov22, aug_cov12


def _trig_trig_cross(mean1, cov1, mean2, cov2, cross, angles) -> torch.Tensor:
    """``Cov[trig(x1), trig(x2)]`` for a pair of correlated Gaussians."""
    angles = list(angles)
    A = len(angles)
    mu = mean1[angles]
    nu = mean2[angles]
    p = torch.diagonal(cov1)[angles]
    q = torch.diagonal(cov2)[angles]
    r = cross[angles][:, angles]                       # (A, A)
    D = torch.exp(-0.5 * (p[:, None] + q[None, :] - 2 * r))
    P = torch.exp(-0.5 * (p[:, None] + q[None, :] + 2 * r))
    dm = mu[:, None] - nu[None, :]
    sm = mu[:, None] + nu[None, :]

    e_ss = 0.5 * (torch.cos(dm) * D - torch.cos(sm) * P)
    e_cc = 0.5 * (torch.cos(dm) * D + torch.cos(sm) * P)
    e_sc = 0.5 * (torch.sin(sm) * P + torch.sin(dm) * D)
    e_cs = 0.5 * (torch.sin(sm) * P - torch.sin(dm) * D)

    # TrigAugmentOp interleaves the harmonics:
    #   row/col 2k = sin(angle_k), 2k+1 = cos(angle_k)
    s1 = torch.exp(-0.5 * p) * torch.sin(mu)
    c1 = torch.exp(-0.5 * p) * torch.cos(mu)
    s2 = torch.exp(-0.5 * q) * torch.sin(nu)
    c2 = torch.exp(-0.5 * q) * torch.cos(nu)
    out = torch.zeros(2 * A, 2 * A, dtype=cov1.dtype, device=cov1.device)
    block_ss = e_ss - torch.outer(s1, s2)
    block_sc = e_sc - torch.outer(s1, c2)
    block_cs = e_cs - torch.outer(c1, s2)
    block_cc = e_cc - torch.outer(c1, c2)
    for k in range(A):
        for m in range(A):
            out[2 * k, 2 * m] = block_ss[k, m]
            out[2 * k, 2 * m + 1] = block_sc[k, m]
            out[2 * k + 1, 2 * m] = block_cs[k, m]
            out[2 * k + 1, 2 * m + 1] = block_cc[k, m]
    # NOTE: do *not* symmetrise.  This is a cross-covariance between two
    # different random vectors and Cov[sin u, cos v] != Cov[cos u, sin v]
    # in general (they differ by sin(mu - nu) e^{-(p+q-2r)/2}); forcing
    # symmetry here silently corrupts trajectories with more than one angle.
    return out
