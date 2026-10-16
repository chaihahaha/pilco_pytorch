"""GP dynamics as a closed-form moment-matching operator.

This is the PyTorch replacement for ``gp/gp0d.py`` (full GP), ``gp/gp1d.py``
(FITC sparse GP) and every ``dMdm/dSdm/dVdm/dMds/...`` derivative the original
files hand-code.  Only the forward moments are written down; autograd supplies
all derivatives, including the second-order ones.

Model
-----
``E`` independent GPs share inputs ``X`` (training states+controls) and have
per-output ARD squared-exponential kernels::

    k_e(x, x') = sf_e^2 exp(-1/2 sum_d (x_d - x'_d)^2 / ell_{e,d}^2)
    A_e        = K_e + sigma_{n,e}^2 I
    beta_e     = A_e^{-1} Y_e

Predictive moments at an uncertain input ``x ~ N(m, S)`` (PILCO eq. 2.34,
``gp/gp0d.py``):

* ``ell_e[n]    = E[k_e(x, X_n)]``                         (Gaussian integral)
* ``M_e         = beta_e^T ell_e``
* ``L_ef[n,n']  = E[k_e(x, X_n) k_f(x, X_n')]``
* ``Cov[f_e, f_f] = delta_ef sf_e^2
                    - delta_ef tr(A_e^{-1} L_ee)
                    + beta_e^T (L_ef - ell_e ell_f^T) beta_f``
* ``C_e = S^{-1} Cov[x, f_e] = -iL_e B_e^{-1} (iL_e g_e)``
  with ``B_e = I + iL_e S iL_e``, ``iL_e = diag(1/ell_e)`` and
  ``g_e = sum_n beta_{e,n} ell_{e,n} (m - X_n)``.

The ``delta_ef tr(...)`` term is absent for ``e != f`` because two different GPs
are conditionally independent given their inputs; the correlation between the
outputs is induced purely by the uncertainty about the shared input.

``L_ef`` derivation
-------------------
The product of two squared-exponential kernels is again Gaussian in ``x``::

    k_e(x,X_n) k_f(x,X_n') = sf_e^2 sf_f^2 exp(-1/2 x^T A x + b^T x + c)

with ``A = L_e + L_f``, ``b = L_e X_n + L_f X_n'`` and
``c = -1/2 (X_n^T L_e X_n + X_n'^T L_f X_n')``, so with
``G = (I + S A)^{-1} S`` and ``d = L_e(X_n - m) + L_f(X_n' - m)``::

    L_ef[n,n'] = sf_e^2 sf_f^2 |I + A S|^{-1/2}
                 exp(k_n^e + k_n'^f + 1/2 d^T G d)

which is exactly ``gp/gp0d.py``'s ``maha`` construction, written in the
natural tensor notation instead of MATLAB/Octave index arithmetic.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..gaussian import logdet_psd, logdet_psd_scaled_diag, symmetrize
from .base import MomentOp, Moments

__all__ = ["GpDynamicsOp"]


def _robust_cholesky(a: torch.Tensor) -> torch.Tensor:
    """Cholesky with progressive jitter (hyper-parameter steps can be wild)."""
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    scale = torch.clamp(torch.diagonal(a).abs().max(), min=1.0)
    last: Exception | None = None
    for k in range(10):
        try:
            return torch.linalg.cholesky(a + scale * 1e-12 * 10 ** k * eye)
        except RuntimeError as exc:  # pragma: no cover - numerical fallback
            last = exc
    raise RuntimeError("Gram matrix not factorisable") from last


class GpDynamicsOp(MomentOp):
    """Sum of ``E`` independent ARD squared-exponential GP regressors.

    Parameters
    ----------
    inputs:
        ``(n, d)`` training inputs (PILCO's ``dynmodel.inputs``).
    targets:
        ``(n, E)`` training targets, i.e. **state differences** for PILCO.
    lengthscales:
        ``(d, E)`` or ``(d,)`` ARD lengthscales.
    signal:
        ``(E,)`` signal standard deviations (not variances).
    noise:
        ``(E,)`` observation-noise standard deviations.
    individual_noise:
        Optional ``(n, E)`` per-datapoint noise VARIANCES (PILCO's ``nigp``),
        used by the sparse/FITC variant.
    """

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor,
                 lengthscales: torch.Tensor, signal: torch.Tensor,
                 noise: torch.Tensor,
                 individual_noise: Optional[torch.Tensor] = None,
                 share_lengthscales: bool = False):
        super().__init__()
        inputs = torch.as_tensor(inputs)
        targets = torch.as_tensor(targets)
        if inputs.ndim != 2 or targets.shape[0] != inputs.shape[0]:
            raise ValueError("inputs and targets must be (n, d) and (n, E)")
        d, E = inputs.shape[1], targets.shape[1]
        ell = torch.as_tensor(lengthscales)
        if ell.numel() == d:
            ell = ell.reshape(d, 1).expand(d, E).clone()
        else:
            ell = ell.reshape(d, E).clone()
        sig = torch.as_tensor(signal).reshape(-1)
        noi = torch.as_tensor(noise).reshape(-1)
        if sig.numel() == 1:
            sig = sig.expand(E).clone()
        if noi.numel() == 1:
            noi = noi.expand(E).clone()

        self.n_in, self.n_out = d, E
        #: When true, one ARD lengthscale per input dimension is shared by all
        #: outputs, halving-ish the hyper-parameter count.  PILCO uses one per
        #: (input, output) pair, which is 24 parameters for cart-pole's 6 inputs
        #: and 4 outputs -- more than the number of data points in the first
        #: episodes.  See ``docs/ARCHITECTURE.md`` section 8.
        self.share_lengthscales = bool(share_lengthscales)
        if self.share_lengthscales:
            ell = ell[:, :1]
        self.register_buffer("inputs", inputs.clone())
        self.register_buffer("targets", targets.clone())
        self.log_lengthscales = nn.Parameter(torch.log(ell.clone()))
        self.log_signal = nn.Parameter(torch.log(sig.clone()))
        self.log_noise = nn.Parameter(torch.log(noi.clone()))
        if individual_noise is None:
            individual_noise = torch.zeros_like(targets)
        self.register_buffer("individual_noise", individual_noise.clone())
        self._cache: dict = {}

    # ------------------------------------------------------------------
    @classmethod
    def from_log_hyp(cls, inputs: torch.Tensor, targets: torch.Tensor,
                     log_hyp: torch.Tensor,
                     individual_noise: Optional[torch.Tensor] = None,
                     share_lengthscales: bool = False) -> "GpDynamicsOp":
        """Build from PILCO's hyper-parameter layout ``(d + 2, E)``.

        Rows ``0..d-1`` are log lengthscales, row ``d`` is the log signal
        standard deviation and row ``d+1`` the log noise standard deviation.
        """
        log_hyp = torch.as_tensor(log_hyp)
        d = log_hyp.shape[0] - 2
        return cls(inputs, targets, torch.exp(log_hyp[:d]),
                   torch.exp(log_hyp[d]), torch.exp(log_hyp[d + 1]),
                   individual_noise=individual_noise,
                   share_lengthscales=share_lengthscales)

    def set_log_hyp(self, value: torch.Tensor) -> None:
        """Overwrite the hyper-parameters from a ``(d + 2, E)`` tensor.

        Used by the learning loop's trust region and best-so-far bookkeeping.
        """
        value = torch.as_tensor(value)
        d = self.n_in
        with torch.no_grad():
            if self.share_lengthscales:
                self.log_lengthscales.copy_(value[:d, :1])
            else:
                self.log_lengthscales.copy_(value[:d])
            self.log_signal.copy_(value[d])
            self.log_noise.copy_(value[d + 1])
        self.invalidate_cache()

    def log_hyp(self) -> torch.Tensor:
        """The hyper-parameters in PILCO's ``(d + 2, E)`` layout."""
        ell = self.log_lengthscales
        if self.share_lengthscales:
            ell = ell.expand(self.n_in, self.n_out)
        return torch.cat([ell, self.log_signal[None, :],
                          self.log_noise[None, :]], dim=0)

    # ------------------------------------------------------------------
    def _log_ell(self, e: int) -> torch.Tensor:
        """Log lengthscales used for output ``e`` (shared or per-output)."""
        if self.share_lengthscales:
            return self.log_lengthscales[:, 0]
        return self.log_lengthscales[:, e]

    @property
    def lengthscales(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscales)

    @property
    def signal(self) -> torch.Tensor:
        return torch.exp(self.log_signal)

    @property
    def noise(self) -> torch.Tensor:
        return torch.exp(self.log_noise)

    def kernel(self, a: torch.Tensor, b: torch.Tensor, e: int) -> torch.Tensor:
        """``k_e`` between rows of ``a`` and ``b``."""
        iell = torch.exp(-self._log_ell(e))
        dist = ((a[:, None, :] - b[None, :, :]) * iell) ** 2
        return (self.signal[e] ** 2) * torch.exp(-0.5 * dist.sum(-1))

    def gram(self, e: int) -> torch.Tensor:
        """``K_e + sigma_e^2 I + diag(nigp_e)``."""
        K = self.kernel(self.inputs, self.inputs, e)
        eye = torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
        return K + (self.noise[e] ** 2) * eye + torch.diag(self.individual_noise[:, e])

    def invalidate_cache(self) -> None:
        """Drop cached Gram inverses (call after changing data or hyper-params)."""
        self._cache = {}

    def _solve_cache(self):
        """Cache ``A_e^{-1}`` and ``beta_e``.

        PILCO optimises the policy with the dynamics hyper-parameters held fixed,
        so this cache is hit for the whole inner L-BFGS loop; it is invalidated
        whenever the buffers or the log-hyper-parameters change.

        When the hyper-parameters *do* require gradients the cache is bypassed:
        caching graph-attached tensors would make the next ``backward()`` fail
        with "trying to backward through the graph a second time", because the
        cached sub-graph would be traversed twice.
        """
        needs_graph = (self.log_lengthscales.requires_grad
                       or self.log_signal.requires_grad
                       or self.log_noise.requires_grad)
        if needs_graph:
            iK, beta = [], []
            for e in range(self.n_out):
                A = self.gram(e)
                iK.append(torch.linalg.inv(A))
                beta.append(torch.linalg.solve(A, self.targets[:, e]))
            return torch.stack(iK), torch.stack(beta, dim=1)

        key = (self.inputs.data_ptr(), self.targets.data_ptr(),
               self.inputs.shape,
               float(self.log_signal.sum()) + float(self.log_noise.sum())
               + float(self.log_lengthscales.sum()))
        cached = self._cache.get("key")
        if cached == key and "iK" in self._cache:
            return self._cache["iK"], self._cache["beta"]
        iK, beta = [], []
        with torch.no_grad():
            for e in range(self.n_out):
                A = self.gram(e)
                iK.append(torch.linalg.inv(A))
                beta.append(torch.linalg.solve(A, self.targets[:, e]))
        iK = torch.stack(iK)
        beta = torch.stack(beta, dim=1)
        self._cache = {"key": key, "iK": iK, "beta": beta}
        return iK, beta

    # ------------------------------------------------------------------
    def nlml(self) -> torch.Tensor:
        """Negative log marginal likelihood summed over outputs.

        Replaces ``gp/hypCurb.py`` + its hand-written derivatives with plain
        autograd.  The Cholesky factorisation is done with progressive jitter so
        that an exploratory hyper-parameter step cannot abort training.
        """
        total = torch.zeros((), dtype=self.inputs.dtype)
        n = self.inputs.shape[0]
        for e in range(self.n_out):
            A = self.gram(e)
            L = _robust_cholesky(A)
            alpha = torch.cholesky_solve(self.targets[:, e].unsqueeze(-1), L).squeeze(-1)
            total = total + 0.5 * (self.targets[:, e] @ alpha)
            total = total + torch.log(torch.diagonal(L)).sum()
            total = total + 0.5 * n * torch.log(torch.tensor(2 * torch.pi, dtype=A.dtype))
        return total

    # ------------------------------------------------------------------
    def moments_batched(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        """Predictive moments at an uncertain input, fully batched (NOT USED).

        Two levels of batching replace the nested Python loops (see the profile in
        ``scripts/diagnostics/profile_hotspots.py``: this call is ~100% of a
        planning step).

        **Status: experimental, not used.**  Compared against ``gp/gp0d.py`` it
        agrees only to ~1e-3 in the predictive covariance, where the loop version
        (``moments``) agrees to ~1e-12, so it fails the validation the operators
        are held to.  The individual pieces (``G``, ``d_e``, ``d_f``, the cross
        term, ``L_ef`` for ``(0, 0)``) were each checked to 1e-13, so the
        discrepancy is somewhere in the ``(E, E)`` assembly that was not tracked
        down.  It is kept here because it is most of the way to a ~3x speedup of
        the planner and someone with more time should finish it.
        """
        d, E = self.n_in, self.n_out
        dtype, device = mean.dtype, mean.device
        eye = torch.eye(d, dtype=dtype, device=device)
        iK, beta = self._solve_cache()

        iell = torch.exp(-self.log_lengthscales.transpose(0, 1))      # (E, d)
        iell2 = iell ** 2
        B = eye + iell[:, :, None] * iell[:, None, :] * cov[None]     # (E, d, d)
        ytilde = (mean[None, None, :] - self.inputs[None, :, :]) * iell[:, None, :]
        sol = torch.linalg.solve(B, ytilde.transpose(1, 2)).transpose(1, 2)
        k_n = -0.5 * (ytilde * ytilde).sum(-1)                        # (E, n)
        logdet_B = 2.0 * torch.log(
            torch.diagonal(torch.linalg.cholesky(symmetrize(B)), dim1=-2, dim2=-1)
        ).sum(-1)                                                     # (E,)
        ell = (self.signal ** 2)[:, None] * torch.exp(
            -0.5 * logdet_B[:, None] - 0.5 * (ytilde * sol).sum(-1))   # (E, n)
        M = (beta.transpose(0, 1) * ell).sum(1)                       # (E,)

        # R[e] = (X - m) L_e  ->  (E, n, d)
        R = (self.inputs[None, :, :] - mean[None, None, :]) * iell2[:, None, :]

        # ---- product-kernel means L_ef[x, y] = E[k_e(x,X) k_f(y,X)] --------
        lam = iell2[:, None, :] + iell2[None, :, :]                   # (E, E, d)
        # cov @ diag(lam): the diagonal multiplies the COLUMN index, i.e. the
        # last axis here.  Getting this transposed was a real bug caught by
        # the loop-vs-batched comparison.
        A = eye[None, None] + cov[None, None, :, :] * lam[:, :, None, :]
        G = torch.linalg.solve(A, cov[None, None].expand(E, E, d, d))
        sqrt_lam = torch.sqrt(lam.clamp(min=0.0))
        scaled = symmetrize(cov[None, None] * sqrt_lam[:, :, :, None]
                            * sqrt_lam[:, :, None, :])                # (E, E, d, d)
        _, log_norm = torch.linalg.slogdet(eye[None, None] + scaled)
        log_norm = -0.5 * log_norm                                    # (E, E)

        RG = torch.einsum("enp,efpq->efnq", R, G)                     # (E, E, n, d)
        d_e = (RG * R[:, None, :, :]).sum(-1)                         # (E, E, n)
        # d_f[e, f, m] = R_f[m] G_ef R_f[m].  Note that m must stay an OUTPUT
        # index: writing this as an einsum "efnq,fmq->efn" silently sums over
        # m instead, which produced a wildly wrong covariance.
        d_f = (RG * R[None, :, :, :]).sum(-1)
        cross = torch.einsum("efnq,fmq->efnm", RG, R)                 # (E, E, n, n)
        quad = 0.5 * (d_e[:, :, :, None] + d_f[:, :, None, :] + 2.0 * cross)

        sf2 = self.signal ** 2
        exponent = (k_n[:, None, :, None] + k_n[None, :, None, :]
                    + quad + log_norm[:, :, None, None])
        Lmat = (sf2[:, None, None, None] * sf2[None, :, None, None]
                * torch.exp(exponent))                                # (E, E, n, n)

        pre = torch.einsum("ne,efnm,mf->ef", beta, Lmat, beta)
        trace_term = torch.einsum("enn,efnn->ef", iK, Lmat)
        pre = pre + torch.diag_embed(
            sf2 - torch.diagonal(trace_term, dim1=0, dim2=1))
        out_cov = symmetrize(pre) - torch.outer(M, M)

        # ---- input-output regression matrix C_e ---------------------------
        g = ((beta * ell.transpose(0, 1)).transpose(0, 1)[:, :, None]
             * (mean[None, None, :] - self.inputs[None, :, :])).sum(1)  # (E, d)
        cross_c = torch.linalg.solve(B, (iell * g).unsqueeze(-1)).squeeze(-1)
        cross = (-iell * cross_c).transpose(0, 1)                     # (d, E)
        return Moments(M, out_cov, cross)

    def moments(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        """Predictive moments at an uncertain input (per-output / per-pair loop).

        This is the *validated* implementation: it agrees with ``gp/gp0d.py`` to
        ~1e-12.  :meth:`moments_batched` is a partially finished attempt to
        replace the loops with batched linear algebra; see its docstring.
        """
        d, E = self.n_in, self.n_out
        dtype, device = mean.dtype, mean.device
        eye = torch.eye(d, dtype=dtype, device=device)
        iK, beta = self._solve_cache()

        ell_cols, k_cols, iell_cols, B_cols = [], [], [], []
        for e in range(E):
            iell = torch.exp(-self._log_ell(e))
            B = eye + (iell[:, None] * iell[None, :]) * cov
            ytilde = (mean[None, :] - self.inputs) * iell
            sol = torch.linalg.solve(B, ytilde.transpose(0, 1)).transpose(0, 1)
            k_n = -0.5 * (ytilde * ytilde).sum(-1)
            ell = (self.signal[e] ** 2) * torch.exp(
                -0.5 * logdet_psd(B) - 0.5 * (ytilde * sol).sum(-1))
            ell_cols.append(ell)
            k_cols.append(k_n)
            iell_cols.append(iell)
            B_cols.append(B)
        ell = torch.stack(ell_cols, dim=1)
        k_n = torch.stack(k_cols, dim=1)
        M = (beta * ell).sum(0)

        R = []
        for e in range(E):
            Le = torch.diag(iell_cols[e] ** 2)
            R.append((self.inputs - mean[None, :]) @ Le)

        rows = []
        for e in range(E):
            row = []
            for f in range(E):
                lam = iell_cols[e] ** 2 + iell_cols[f] ** 2
                A = torch.diag(lam)
                G = torch.linalg.solve(eye + cov @ A, cov)
                log_norm = -0.5 * logdet_psd_scaled_diag(cov, lam, factor=1.0)
                d_e = torch.einsum("nd,de,ne->n", R[e], G, R[e])
                d_f = torch.einsum("nd,de,ne->n", R[f], G, R[f])
                cross = torch.einsum("nd,de,me->nm", R[e], G, R[f])
                quad = d_e[:, None] + d_f[None, :] + 2.0 * cross
                Lmat = (self.signal[e] ** 2) * (self.signal[f] ** 2) * torch.exp(
                    k_n[:, e][:, None] + k_n[:, f][None, :] + 0.5 * quad + log_norm)
                pre = beta[:, e] @ Lmat @ beta[:, f]
                if e == f:
                    pre = pre + (self.signal[e] ** 2) - (iK[e] * Lmat).sum()
                row.append(pre)
            rows.append(torch.stack(row))
        out_cov = torch.stack(rows) - torch.outer(M, M)

        cross_cols = []
        for e in range(E):
            iell = iell_cols[e]
            g = ((beta[:, e] * ell[:, e])[:, None]
                 * (mean[None, :] - self.inputs)).sum(0)
            cross_cols.append(-iell * torch.linalg.solve(B_cols[e], iell * g))
        cross = torch.stack(cross_cols, dim=1)
        return Moments(M, symmetrize(out_cov), cross)
