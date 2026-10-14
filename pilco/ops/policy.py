"""The RBF (squared-exponential) controller as a closed-form moment operator.

PILCO's controller is the *mean* of a GP whose "training set" is a small set of
learnable pseudo-inputs.  With a squared-exponential kernel and fixed signal
variance ``1`` and noise ``sigma_n`` this is exactly a regularised RBF network::

    u(x) = sum_n beta_n * k(x, X_n),      k(x, X_n) = exp(-1/2 (x-X_n)^T L (x-X_n))
    beta = (K + sigma_n^2 I)^{-1} T,      K[n,n']  = k(X_n, X_n')
    L = diag(1/ell^2)

with learnable pseudo-inputs ``X`` (``centers``), pseudo-targets ``T`` and
ARD lengthscales ``ell``.  This mirrors ``control/congp.py`` + ``gp/gp2.py`` in
the NumPy reference (which fixes signal variance to 1 and noise to 0.01).

Closed-form moments under ``x ~ N(m, S)``
----------------------------------------
With ``delta_n = m - X_n`` and the Gaussian integral
``E[exp(b^T y - 1/2 y^T A y)] = |I+A S|^{-1/2} exp(1/2 b^T (I+S A)^{-1} S b)``:

* ``ell_n = E[k(x,X_n)] = |I+L S|^{-1/2} exp(-1/2 delta_n^T L (I+S L)^{-1} delta_n)``
* ``k_n   = -1/2 delta_n^T L delta_n``   (kernel exponent evaluated at the mean)
* ``Lmat[n,n'] = E[k(x,X_n) k(x,X_n')]
                = |I+2 L S|^{-1/2} exp(k_n + k_n' + 1/2 u^T P u)``
  with ``u = X_n + X_n' - 2m`` and ``P = L (I+2 S L)^{-1} S L``.
* ``M          = beta @ ell``
* ``Cov(u)     = beta (Lmat - ell ell^T) beta^T``
* ``C = S^{-1} Cov[x,u] = -(I + L S)^{-1} L @ sum_n beta_n delta_n ell_n``
  (Stein's lemma).

Composition with the saturation nonlinearity is handled by splicing
:class:`~pilco.ops.fourier.SaturationOp` afterwards, which is simpler and more
accurate than the original ``conCat`` hand-derived chain rule.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..gaussian import logdet_psd_scaled_diag, symmetrize
from .base import MomentOp, Moments

__all__ = ["RbfPolicyOp"]


class RbfPolicyOp(MomentOp):
    """Regularised RBF network with ARD lengthscales.

    Parameters
    ----------
    centers:
        ``(N, n_in)`` pseudo-inputs (PILCO's ``policy.p.inputs``).
    targets:
        ``(N, n_out)`` pseudo-targets (``policy.p.targets``).
    lengthscales:
        ``(n_in,)`` ARD lengthscales (PILCO stores the log of these).
    noise:
        Regularisation ``sigma_n`` used when forming ``beta``.
    signal:
        Kernel amplitude (PILCO fixes it to 1).
    """

    def __init__(self, centers: torch.Tensor, targets: torch.Tensor,
                 lengthscales: torch.Tensor, noise: float = 0.01,
                 signal: float = 1.0):
        super().__init__()
        centers = torch.as_tensor(centers)
        targets = torch.as_tensor(targets)
        if centers.ndim != 2:
            raise ValueError("centers must be (N, n_in)")
        if targets.shape[0] != centers.shape[0]:
            raise ValueError("targets and centers must have the same length")
        ell = torch.as_tensor(lengthscales).reshape(-1)
        if ell.numel() != centers.shape[1]:
            raise ValueError("lengthscales must have one entry per input dim")

        self.n_in = centers.shape[1]
        self.n_out = targets.shape[1]
        self.noise = float(noise)
        self.signal = float(signal)
        self.centers = nn.Parameter(centers.clone())
        self.targets = nn.Parameter(targets.clone())
        self.log_lengthscales = nn.Parameter(torch.log(ell.clone()))

    # ------------------------------------------------------------------
    @property
    def lengthscales(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscales)

    def precision(self) -> torch.Tensor:
        """``L = diag(1/ell^2)``."""
        iell = torch.exp(-self.log_lengthscales)
        return torch.diag(iell ** 2)

    def kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Squared-exponential kernel ``k(a_i, b_j)`` with ARD lengthscales."""
        iell2 = torch.exp(-2.0 * self.log_lengthscales)
        d = a[:, None, :] - b[None, :, :]
        return self.signal * torch.exp(-0.5 * (d ** 2 * iell2).sum(-1))

    def beta(self) -> torch.Tensor:
        """``(K + sigma_n^2 I)^{-1} T``, shape ``(N, n_out)``."""
        K = self.kernel(self.centers, self.centers)
        A = K + (self.noise ** 2) * torch.eye(
            K.shape[0], dtype=K.dtype, device=K.device)
        return torch.linalg.solve(A, self.targets)

    # ------------------------------------------------------------------
    def moments(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        L = self.precision()                                   # (d, d)
        beta = self.beta()                                     # (N, E)
        delta = mean[None, :] - self.centers                   # (N, d)
        d = self.n_in
        eye = torch.eye(d, dtype=mean.dtype, device=mean.device)

        B1 = eye + cov @ L                                     # I + S L
        diag_L = torch.diagonal(L)
        log_det_B1 = logdet_psd_scaled_diag(cov, diag_L, factor=1.0)

        # k_n = -1/2 delta_n^T L delta_n
        Ldelta = delta @ L                                     # (N, d)
        k_n = -0.5 * (delta * Ldelta).sum(-1)

        # ell_n = |I + L S|^{-1/2} exp(-1/2 delta^T L (I+S L)^{-1} delta)
        # det(I+L S) = det(I+S L) and  L (I+S L)^{-1} = (I+L S)^{-1} L
        solved = torch.linalg.solve(B1, delta.transpose(0, 1)).transpose(0, 1)
        quad = -0.5 * (Ldelta * solved).sum(-1)
        ell = self.signal * torch.exp(-0.5 * log_det_B1 + quad)   # (N,)

        # Lmat[n,n'] = |I + 2 L S|^{-1/2} exp(k_n + k_n' + 1/2 u^T P u)
        B2 = eye + 2.0 * (cov @ L)                             # I + 2 S L
        log_det_B2 = logdet_psd_scaled_diag(cov, diag_L, factor=2.0)
        P = L @ torch.linalg.solve(B2, cov) @ L
        u = (self.centers[:, None, :] + self.centers[None, :, :]
             - 2.0 * mean[None, None, :])
        quad2 = torch.einsum("nmd,de,nme->nm", u, P, u)
        Lmat = (self.signal ** 2) * torch.exp(
            k_n[:, None] + k_n[None, :] + 0.5 * quad2 - 0.5 * log_det_B2)

        M = beta.transpose(0, 1) @ ell                          # (E,)
        inner = Lmat - torch.outer(ell, ell)
        out_cov = beta.transpose(0, 1) @ inner @ beta           # (E, E)

        # C = -(I + L S)^{-1} L @ sum_n beta_n delta_n ell_n
        #   = -(I + S L)^{-T} L @ (beta^T (delta * ell))^T
        weighted = (beta.transpose(0, 1)[:, :, None]
                    * (ell[:, None] * delta)[None, :, :])       # (E, N, d)
        g = weighted.sum(1)                                     # (E, d)
        rhs = (g @ L).transpose(0, 1)                           # (d, E) = L g^T
        cross = -torch.linalg.solve(B1.transpose(0, 1), rhs)

        return Moments(M, symmetrize(out_cov), cross)
