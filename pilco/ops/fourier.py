"""Closed-form Gaussian moments of sine/cosine harmonics.

This single module replaces four hand-derived operators of the original code
base:

===============================  ==================================================
original                         this module
===============================  ==================================================
``util/gTrig.py``                :class:`TrigAugmentOp`
``util/gSin.py`` + ``util/gSat.py``  :class:`SaturationOp`
``loss/lossSat.py`` (partly)     :class:`~pilco.ops.cost.SaturatingCostOp`
===============================  ==================================================

The mathematics
---------------
For ``x ~ N(m, S)`` and a harmonic ``h(x) = sin(f * x_i)`` (or ``cos``) we need

* ``E[h(x)]``,
* ``Cov[h(x), h'(x)]`` for every pair of harmonics,
* ``Cov[x_j, h(x)]`` (PILCO's ``C``).

All three have textbook closed forms because for jointly Gaussian ``(u, v)``
with means ``(mu, nu)`` and covariance ``[[a, c], [c, b]]``::

    E[sin u]            = e^{-a/2} sin(mu)
    E[sin u sin v]      = 1/2 [ cos(mu-nu) e^{-(a+b-2c)/2}
                               - cos(mu+nu) e^{-(a+b+2c)/2} ]
    E[cos u cos v]      = 1/2 [ cos(mu-nu) e^{-(a+b-2c)/2}
                               + cos(mu+nu) e^{-(a+b+2c)/2} ]
    E[sin u cos v]      = 1/2 [ sin(mu+nu) e^{-(a+b+2c)/2}
                               + sin(mu-nu) e^{-(a+b-2c)/2} ]
    Cov[x_j, sin(f x_i)] = f S_{ji} e^{-f^2 S_{ii}/2} cos(f m_i)     (Stein)
    Cov[x_j, cos(f x_i)] = -f S_{ji} e^{-f^2 S_{ii}/2} sin(f m_i)    (Stein)

Treating each harmonic as its own random variable with mean ``f mu_i`` and
covariance ``f^2 S_{ii}`` makes frequency-3 terms such as
``sin(3 x_i)`` fall out of the same formulas, which is exactly how the
original ``gSat`` is built by stacking ``[m; 3m]`` and ``blkdiag(S, 9S)``.

Why this matters for the rewrite
--------------------------------
The original code hand-differentiates every one of the formulas above
(``dMdm, dVdm, dCdm, dMdv, dVdv, dCdv``), including the Fortran-order
``reshape`` gymnastics needed to match MATLAB's ``vec`` convention.  We simply
evaluate the formulas with torch tensors; autograd produces all derivatives,
including second-order ones (needed for the Hessian of the value function).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from ..gaussian import symmetrize
from .base import MomentOp, Moments

__all__ = ["Harmonic", "SIN", "COS", "TrigAugmentOp", "SaturationOp", "FourierOp"]

SIN = "sin"
COS = "cos"


class Harmonic:
    """One basis function ``scale * trig(freq * x[index])``.

    Parameters
    ----------
    index:
        Coordinate of the operator's input vector the harmonic reads.
    freq:
        Frequency ``f``; ``1`` for ``sin(x)``, ``3`` for ``sin(3x)``.
    kind:
        ``"sin"`` or ``"cos"``.
    scale:
        Amplitude applied to this harmonic.
    """

    __slots__ = ("index", "freq", "kind", "scale")

    def __init__(self, index: int, freq: float = 1.0, kind: str = SIN,
                 scale: float = 1.0):
        if kind not in (SIN, COS):
            raise ValueError(f"kind must be {SIN!r} or {COS!r}, got {kind!r}")
        self.index = int(index)
        self.freq = float(freq)
        self.kind = kind
        self.scale = float(scale)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Harmonic(index={self.index}, freq={self.freq}, "
                f"kind={self.kind!r}, scale={self.scale})")


def _harmonics_moments(mean: torch.Tensor, cov: torch.Tensor,
                       harmonics: Sequence[Harmonic]) -> Moments:
    """Joint moments of a list of (possibly scaled) harmonics.

    Returns moments of the *scaled* harmonic vector, i.e. the operator output
    is ``[h_0(x), ..., h_{K-1}(x)]`` directly.
    """
    K = len(harmonics)
    dtype, device = mean.dtype, mean.device
    if K == 0:
        zeros = torch.zeros(0, dtype=dtype, device=device)
        return Moments(zeros, torch.zeros(0, 0, dtype=dtype, device=device),
                       torch.zeros(mean.numel(), 0, dtype=dtype, device=device))

    idx = torch.tensor([h.index for h in harmonics], dtype=torch.long, device=device)
    freq = torch.tensor([h.freq for h in harmonics], dtype=dtype, device=device)
    scale = torch.tensor([h.scale for h in harmonics], dtype=dtype, device=device)
    is_sin = torch.tensor([h.kind == SIN for h in harmonics], dtype=torch.bool, device=device)

    # Effective scalar Gaussian of each harmonic argument:  f * x_index
    mu = freq * mean[idx]                                     # (K,)
    vv = (freq ** 2) * torch.diagonal(cov)[idx]                # (K,)
    cross_v = (freq[:, None] * freq[None, :]) * cov[idx][:, idx]  # (K, K)

    vd = vv[:, None] + vv[None, :] - 2 * cross_v              # Var(u - v)
    vs = vv[:, None] + vv[None, :] + 2 * cross_v              # Var(u + v)
    ed = torch.exp(-vd / 2)
    es = torch.exp(-vs / 2)
    dm = mu[:, None] - mu[None, :]
    sm = mu[:, None] + mu[None, :]

    e_ss = 0.5 * (torch.cos(dm) * ed - torch.cos(sm) * es)
    e_cc = 0.5 * (torch.cos(dm) * ed + torch.cos(sm) * es)
    e_sc = 0.5 * (torch.sin(sm) * es + torch.sin(dm) * ed)
    e_cs = 0.5 * (torch.sin(sm) * es - torch.sin(dm) * ed)

    # assemble the raw second-moment matrix for the sin/cos mix
    sin_sin = is_sin[:, None] & is_sin[None, :]
    cos_cos = (~is_sin)[:, None] & (~is_sin)[None, :]
    sin_cos = is_sin[:, None] & (~is_sin)[None, :]
    cos_sin = (~is_sin)[:, None] & is_sin[None, :]
    second = (e_ss * sin_sin + e_cc * cos_cos
              + e_sc * sin_cos + e_cs * cos_sin)

    decay = torch.exp(-vv / 2)
    first = scale * decay * torch.where(is_sin, torch.sin(mu), torch.cos(mu))

    second = (scale[:, None] * scale[None, :]) * second
    out_mean = first
    out_cov = symmetrize(second - torch.outer(out_mean, out_mean))

    # Cov[x_j, h_k] via Stein's lemma:
    #   Cov[x_j, sin(f x_i)] =  f S_{ji} e^{-f^2 S_ii/2} cos(f m_i)
    #   Cov[x_j, cos(f x_i)] = -f S_{ji} e^{-f^2 S_ii/2} sin(f m_i)
    # so Cov[x, h_k] = S[:, i_k] * coeff_k and therefore, because the operator
    # reports C = S^{-1} Cov[x, h] (PILCO's convention),
    #   C[i_k, k] = coeff_k   and zero elsewhere.
    d_trig = torch.where(is_sin, torch.cos(mu), -torch.sin(mu))     # (K,)
    coeff = scale * freq * decay * d_trig                            # (K,)
    out_cross = torch.zeros(mean.numel(), K, dtype=dtype, device=device)
    out_cross = out_cross.index_put((idx, torch.arange(K, device=device)), coeff)
    return Moments(out_mean, out_cov, out_cross)


class FourierOp(MomentOp):
    """Generic operator defined by a list of harmonics and an optional readout.

    Parameters
    ----------
    n_in:
        Dimension of the input Gaussian.
    harmonics:
        Basis harmonics (see :class:`Harmonic`).
    readout:
        Optional ``(n_out, len(harmonics))`` matrix combining the harmonics
        linearly.  Defaults to the identity, so the output vector *is* the
        harmonic vector.
    """

    def __init__(self, n_in: int, harmonics: Sequence[Harmonic],
                 readout: torch.Tensor | None = None):
        super().__init__()
        self.n_in = int(n_in)
        self.harmonics = list(harmonics)
        K = len(self.harmonics)
        if readout is None:
            readout = torch.eye(K, dtype=torch.get_default_dtype())
        self.n_out = readout.shape[0]
        self.register_buffer("readout", readout.clone())

    def moments(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        base = _harmonics_moments(mean, cov, self.harmonics)
        R = self.readout
        if R.shape[1] != base.n_out:
            raise ValueError(f"readout has {R.shape[1]} columns but there are "
                             f"{base.n_out} harmonics")
        m = R @ base.mean
        S = symmetrize(R @ base.cov @ R.transpose(0, 1))
        cross = base.cross @ R.transpose(0, 1)
        return Moments(m, S, cross)


class TrigAugmentOp(MomentOp):
    """PILCO's ``gTrig``: augment the state with ``sin``/``cos`` of angles.

    ``x -> [sin(x_a1), cos(x_a1), sin(x_a2), cos(x_a2), ...]``

    Parameters
    ----------
    angle_indices:
        Coordinates of the *input* Gaussian that hold angles.
    scale:
        Optional per-angle amplitude (PILCO's ``e``).
    """

    def __init__(self, angle_indices: Sequence[int], n_in: int | None = None,
                 scale: Sequence[float] | None = None):
        super().__init__()
        angles = [int(a) for a in angle_indices]
        self.angle_indices = angles
        self.n_out = 2 * len(angles)
        self.n_in = int(n_in) if n_in is not None else (max(angles) + 1 if angles else 0)
        if scale is None:
            scale = [1.0] * len(angles)
        harmonics: list[Harmonic] = []
        for a, s in zip(angles, scale):
            harmonics.append(Harmonic(a, 1.0, SIN, s))
            harmonics.append(Harmonic(a, 1.0, COS, s))
        self.harmonics = harmonics

    def moments(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        return _harmonics_moments(mean, cov, self.harmonics)


class SaturationOp(MomentOp):
    """PILCO's ``gSat``: ``u -> maxU * (9 sin(u) + sin(3u)) / 8``.

    ``u`` is the *unsaturated* control signal.  The cubic-ish odd function
    ``(9 sin u + sin 3u)/8`` is a smooth stand-in for ``tanh`` that keeps the
    moment integrals closed form.
    """

    def __init__(self, n_in: int, channel_indices: Sequence[int],
                 max_u: Sequence[float] | torch.Tensor):
        super().__init__()
        self.n_in = int(n_in)
        self.channel_indices = [int(c) for c in channel_indices]
        mu = torch.as_tensor(max_u, dtype=torch.get_default_dtype()).reshape(-1)
        if mu.numel() != len(self.channel_indices):
            raise ValueError("max_u must have one entry per channel")
        self.n_out = len(self.channel_indices)
        self.harmonics = []
        for c in self.channel_indices:
            self.harmonics.append(Harmonic(c, 1.0, SIN, 1.0))
            self.harmonics.append(Harmonic(c, 3.0, SIN, 1.0))
        R = torch.zeros(self.n_out, 2 * self.n_out, dtype=mu.dtype)
        for i, m in enumerate(mu):
            R[i, 2 * i] = 9.0 / 8.0 * m
            R[i, 2 * i + 1] = 1.0 / 8.0 * m
        self.register_buffer("readout", R)

    def moments(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        base = _harmonics_moments(mean, cov, self.harmonics)
        R = self.readout
        m = R @ base.mean
        S = symmetrize(R @ base.cov @ R.transpose(0, 1))
        cross = base.cross @ R.transpose(0, 1)
        return Moments(m, S, cross)
