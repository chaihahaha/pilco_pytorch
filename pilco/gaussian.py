"""Gaussian distributions and small linear-algebra helpers.

Everything in :mod:`pilco` is expressed as *moment matching*: we only ever keep
the first two moments (mean and covariance) of a Gaussian.  This module holds
the tiny container used to pass those moments around plus the numerically safe
linear-algebra primitives (Cholesky-based solve, determinant, log-determinant,
PSD projection) that the rest of the package builds on.

Design notes
------------
* ``dtype`` is ``torch.float64`` by default.  PILCO's analytic moment matching
  is numerically delicate (exponentials of covariance terms, determinants of
  near-singular matrices); single precision regularly produces negative
  variances.
* Covariance matrices are always symmetrised before use.  The analytic
  formulas accumulate tiny asymmetries which otherwise make Cholesky fail.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

import torch

__all__ = [
    "Gaussian", "symmetrize", "jitter", "safe_cholesky", "solve_psd",
    "logdet_psd", "logdet", "logdet_psd_scaled_diag", "nearest_psd",
    "quadratic_form",
]

#: Default floating point precision for the whole package.
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# linear algebra helpers
# ---------------------------------------------------------------------------
def symmetrize(a: torch.Tensor) -> torch.Tensor:
    """Return the symmetric part ``(a + a.T) / 2`` of a square matrix."""
    return 0.5 * (a + a.transpose(-1, -2))


def jitter(a: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Add a small ridge to the diagonal, scaled by the matrix magnitude."""
    a = symmetrize(a)
    scale = torch.diagonal(a, dim1=-2, dim2=-1).abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(scale, min=1.0)
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    return a + eps * scale.unsqueeze(-1) * eye


def safe_cholesky(a: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Cholesky factor with progressive jitter.

    Analytic PILCO routinely produces covariance matrices that are PSD but
    numerically indefinite.  We first symmetrise, then retry with increasing
    ridges before giving up and projecting to the nearest PSD matrix.
    """
    a = symmetrize(a)
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    for k in range(8):
        try:
            return torch.linalg.cholesky(a + (eps * 10.0 ** k) * eye)
        except RuntimeError:
            continue
    return torch.linalg.cholesky(nearest_psd(a) + 1e-6 * eye)


def solve_psd(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Solve ``a x = b`` for a symmetric PSD ``a`` via Cholesky."""
    L = safe_cholesky(a, eps=eps)
    return torch.cholesky_solve(b, L)


def logdet_psd(a: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """``log det a`` for a *symmetric* PSD matrix."""
    L = safe_cholesky(a, eps=eps)
    return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)


def logdet(a: torch.Tensor) -> torch.Tensor:
    """``log det a`` for a general (possibly non-symmetric) matrix.

    Needed because PILCO's covariance algebra constantly produces products such
    as ``S @ W`` with symmetric ``S`` and ``W``: the product is *not* symmetric,
    yet its determinant (and that of ``I + S W``) is the one the Gaussian
    integrals produce.  Silently symmetrising such a matrix gives a different
    determinant -- a real bug that the NumPy reference has in
    ``loss/lossSat.py``-adjacent code paths.
    """
    return torch.linalg.slogdet(a)[1]


def logdet_psd_scaled_diag(cov: torch.Tensor, diag_scale: torch.Tensor,
                           factor: float = 1.0) -> torch.Tensor:
    """``log det(I + factor * cov @ diag(diag_scale))`` for symmetric ``cov``.

    Computed as ``log det(I + factor * D^{1/2} cov D^{1/2})`` with
    ``D = diag(diag_scale)``, which is symmetric and therefore safe to
    Cholesky-factorise.
    """
    s = torch.sqrt(torch.clamp(factor * diag_scale, min=0.0))
    a = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device) \
        + symmetrize(cov * s[:, None] * s[None, :])
    return logdet_psd(a)


def quadratic_form(a: torch.Tensor, x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """``x^T a^{-1} x`` for a symmetric PSD ``a`` and ``x`` of shape ``(..., n)``."""
    y = solve_psd(a, x.unsqueeze(-1), eps=eps).squeeze(-1)
    return (x * y).sum(-1)


def nearest_psd(a: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    """Project a symmetric matrix onto the PSD cone (eigenvalue clipping)."""
    a = symmetrize(a)
    w, v = torch.linalg.eigh(a)
    w = torch.clamp(w, min=floor)
    return (v * w.unsqueeze(-2)) @ v.transpose(-1, -2)


# ---------------------------------------------------------------------------
# Gaussian container
# ---------------------------------------------------------------------------
@dataclass
class Gaussian:
    """A multivariate normal ``N(mean, cov)``.

    Only ``mean`` and ``cov`` are stored; the full covariance is retained (this
    is PILCO's central approximation: correlations between states at different
    times are kept explicitly rather than being discarded, which is what makes
    the cumulative-cost variance computable in closed form).
    """

    mean: torch.Tensor          # (n,)
    cov: torch.Tensor           # (n, n)

    # -- construction ------------------------------------------------------
    def __post_init__(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError(f"mean must be 1-D, got shape {tuple(self.mean.shape)}")
        if self.cov.shape != (self.mean.numel(), self.mean.numel()):
            raise ValueError(
                f"cov shape {tuple(self.cov.shape)} does not match "
                f"mean size {self.mean.numel()}"
            )

    @classmethod
    def isotropic(cls, dim: int, std: float = 1.0, dtype=DTYPE) -> "Gaussian":
        return cls(torch.zeros(dim, dtype=dtype),
                   torch.eye(dim, dtype=dtype) * std ** 2)

    @classmethod
    def from_std(cls, mean: Sequence[float], std: Sequence[float]) -> "Gaussian":
        m = torch.as_tensor(mean, dtype=DTYPE)
        s = torch.as_tensor(std, dtype=DTYPE)
        return cls(m, torch.diag(s ** 2))

    # -- basic properties --------------------------------------------------
    @property
    def dim(self) -> int:
        return self.mean.numel()

    def marginal(self, idx) -> "Gaussian":
        """Marginal distribution over a subset of coordinates."""
        idx = _idx(idx)
        return Gaussian(self.mean[idx], self.cov[idx][:, idx])

    def conditional(self, idx_given, idx_query, value=None) -> "Gaussian":
        """``p(query | given)`` of the joint Gaussian ``self``.

        ``value`` defaults to the mean of the conditioning block, i.e. the
        regression of ``query`` onto ``given``.
        """
        g, q = _idx(idx_given), _idx(idx_query)
        S_gg = self.cov[g][:, g]
        S_gq = self.cov[g][:, q]
        S_qq = self.cov[q][:, q]
        if value is None:
            value = self.mean[g]
        gain = solve_psd(S_gg, S_gq)                       # (|g|, |q|)
        mean = self.mean[q] + gain.transpose(-1, -2) @ (value - self.mean[g])
        cov = S_qq - S_gq.transpose(-1, -2) @ gain
        return Gaussian(mean, symmetrize(cov))

    def entropy(self) -> torch.Tensor:
        n = self.dim
        return 0.5 * (n * (1.0 + torch.log(torch.tensor(2 * torch.pi, dtype=self.mean.dtype)))
                      + logdet_psd(self.cov))

    def replace(self, **kwargs) -> "Gaussian":
        return replace(self, **kwargs)

    def clone(self) -> "Gaussian":
        return Gaussian(self.mean.clone(), self.cov.clone())

    # -- conveniences ------------------------------------------------------
    def std(self, idx=None) -> torch.Tensor:
        d = torch.diagonal(self.cov)
        return torch.sqrt(torch.clamp(d if idx is None else d[_idx(idx)], min=0.0))

    def __len__(self) -> int:
        return self.dim


def _idx(idx) -> torch.Tensor:
    if isinstance(idx, torch.Tensor):
        return idx.long()
    if isinstance(idx, (int,)):
        return torch.tensor([idx], dtype=torch.long)
    return torch.as_tensor(list(idx), dtype=torch.long)
