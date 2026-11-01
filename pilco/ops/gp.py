"""GP dynamics as a closed-form moment-matching operator.

The model is **GPyTorch**: one :class:`gpytorch.models.ExactGP` per output
dimension, each with a zero mean and an ARD squared-exponential kernel.  GPyTorch
owns the kernel, the likelihood, the marginal likelihood and the parameter
bookkeeping; this module adds the one thing GPyTorch does not do -- the
**moment-matched prediction at an uncertain input**.

Model
-----
``E`` independent GPs share inputs ``X`` and have per-output ARD kernels::

    k_e(x, x') = sf_e^2 exp(-1/2 sum_d (x_d - x'_d)^2 / ell_{e,d}^2)
    A_e        = K_e + sigma_{n,e}^2 I
    beta_e     = A_e^{-1} Y_e

Predictive moments at ``x ~ N(m, S)`` (PILCO eq. 2.34, ``gp/gp0d.py``):

* ``phi_e[n]    = E[k_e(x, X_n)]``                       (Gaussian integral)
* ``M_e         = beta_e^T phi_e``
* ``Psi_ef[n,m] = E[k_e(x, X_n) k_f(x, X_m)]``
* ``Cov[f_e, f_f] = delta_ef sf_e^2
                    - delta_ef tr(A_e^{-1} Psi_ee)
                    + beta_e^T (Psi_ef - phi_e phi_f^T) beta_f``
* ``C_e = S^{-1} Cov[x, f_e]``

The ``delta_ef tr(...)`` term is absent for ``e != f`` because two different GPs
are conditionally independent given their input; all output correlation comes
from the uncertainty about that shared input.

Why GPyTorch here
-----------------
Everything *around* the moment integrals used to be hand-written: the Gram
matrix, a jittered Cholesky, the log-determinant, the inverse, ``beta``, and a
manual cache keyed on a checksum of the hyper-parameters.  All of that is now
``covar_module``, ``likelihood`` and ``ExactMarginalLogLikelihood``, and the
cache reduces to "does any hyper-parameter need a gradient?" -- see
:meth:`posterior`.

Parameterisation
----------------
GPyTorch normally maps an unconstrained ``raw_*`` parameter to a positive value
with **softplus**.  PILCO optimises in log space and the learning loop's
behaviour (including the reproduction of the NumPy reference) depends on those
coordinates, so the constraint is replaced by ``exp``/``log``.  Consequences:

===============  ============================  =============================
quantity         GPyTorch raw parameter        relation to PILCO's ``hyp``
===============  ============================  =============================
lengthscale      ``raw_lengthscale`` (1, d)    ``raw == log(ell)``
signal variance  ``raw_outputscale`` ()        ``raw == 2 log(sf)``
noise variance   ``raw_noise`` (1,)            ``raw == 2 log(sigma_n)``
===============  ============================  =============================

The factors of two are only there because GPyTorch stores *variances* where PILCO
stores standard deviations; :attr:`log_signal` and :attr:`log_noise` undo them.
"""
from __future__ import annotations

from typing import Optional

import gpytorch
import torch
import torch.nn as nn

from ..gaussian import logdet_psd, logdet_psd_scaled_diag, symmetrize
from .base import MomentOp, Moments

__all__ = ["GpDynamicsOp"]

#: GPyTorch's default constraint is softplus; PILCO works in log coordinates.
_LOG_POSITIVE = gpytorch.constraints.Positive(transform=torch.exp,
                                              inv_transform=torch.log)


class _SingleOutputGP(gpytorch.models.ExactGP):
    """``y = f(x) + eps`` with a zero mean and an ARD squared-exponential kernel."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        super().__init__(inputs, targets,
                         gpytorch.likelihoods.GaussianLikelihood(
                             noise_constraint=_LOG_POSITIVE))
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inputs.shape[1],
                                       lengthscale_constraint=_LOG_POSITIVE),
            outputscale_constraint=_LOG_POSITIVE)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class GpDynamicsOp(MomentOp):
    """``E`` independent ARD squared-exponential GP regressors, one per output.

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
    share_lengthscales:
        Share one ARD lengthscale across all outputs.  Off by default: PILCO uses
        one per (input, output) pair, which is 24 parameters for cart-pole's 6
        inputs and 4 outputs -- more than the number of data points in the first
        episodes.  Measured to hurt; see ``docs/ARCHITECTURE.md`` section 8.
    """

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor,
                 lengthscales: torch.Tensor, signal: torch.Tensor,
                 noise: torch.Tensor,
                 share_lengthscales: bool = False):
        super().__init__()
        inputs = torch.as_tensor(inputs)
        targets = torch.as_tensor(targets)
        if inputs.ndim != 2 or targets.shape[0] != inputs.shape[0]:
            raise ValueError("inputs and targets must be (n, d) and (n, E)")
        d, E = inputs.shape[1], targets.shape[1]
        self.n_in, self.n_out = d, E
        self.share_lengthscales = bool(share_lengthscales)

        self.register_buffer("inputs", inputs.clone())
        self.register_buffer("targets", targets.clone())
        self.models = nn.ModuleList(
            [_SingleOutputGP(self.inputs, self.targets[:, e]) for e in range(E)])
        if self.share_lengthscales:
            kernel = self.models[0].covar_module.base_kernel
            for model in self.models[1:]:
                model.covar_module.base_kernel = kernel

        ell = torch.as_tensor(lengthscales).reshape(d, -1)
        ell = ell.expand(d, E) if ell.shape[1] == 1 else ell
        self.invalidate_cache()
        self.set_log_hyp(torch.cat([
            torch.log(ell),
            torch.log(torch.as_tensor(signal).reshape(1, -1).expand(1, E)),
            torch.log(torch.as_tensor(noise).reshape(1, -1).expand(1, E)),
        ], dim=0))

    # ------------------------------------------------------------------
    # construction / hyper-parameter plumbing
    # ------------------------------------------------------------------
    @classmethod
    def from_log_hyp(cls, inputs: torch.Tensor, targets: torch.Tensor,
                     log_hyp: torch.Tensor,
                     share_lengthscales: bool = False) -> "GpDynamicsOp":
        """Build from PILCO's hyper-parameter layout ``(d + 2, E)``.

        Rows ``0..d-1`` are log lengthscales, row ``d`` is the log signal
        standard deviation and row ``d+1`` the log noise standard deviation.
        """
        log_hyp = torch.as_tensor(log_hyp)
        d = log_hyp.shape[0] - 2
        return cls(inputs, targets, torch.exp(log_hyp[:d]),
                   torch.exp(log_hyp[d]), torch.exp(log_hyp[d + 1]),
                   share_lengthscales=share_lengthscales)

    def set_log_hyp(self, value: torch.Tensor) -> None:
        """Overwrite the hyper-parameters from a ``(d + 2, E)`` tensor.

        Used by the learning loop's trust region and best-so-far bookkeeping.
        """
        value = torch.as_tensor(value)
        d = self.n_in
        with torch.no_grad():
            for e, model in enumerate(self.models):
                if e == 0 or not self.share_lengthscales:
                    model.covar_module.base_kernel.raw_lengthscale.copy_(
                        value[:d, e].reshape(1, -1))
                model.covar_module.raw_outputscale.copy_(2.0 * value[d, e])
                model.likelihood.raw_noise.copy_(
                    (2.0 * value[d + 1, e]).reshape(1))
        self.invalidate_cache()

    def log_hyp(self) -> torch.Tensor:
        """The hyper-parameters in PILCO's ``(d + 2, E)`` layout."""
        return torch.cat([self.log_lengthscales, self.log_signal[None],
                          self.log_noise[None]], dim=0)

    def hyperparameters(self) -> list[torch.Tensor]:
        """The leaf tensors the learning loop optimises (deduplicated)."""
        out: list[torch.Tensor] = []
        for model in self.models:
            for p in (model.covar_module.base_kernel.raw_lengthscale,
                      model.covar_module.raw_outputscale,
                      model.likelihood.raw_noise):
                if not any(p is q for q in out):
                    out.append(p)
        return out

    # -- views on the GPyTorch parameters, in PILCO's coordinates -----------
    def _log_ell(self, e: int) -> torch.Tensor:
        """Log lengthscales used for output ``e``."""
        return self.models[e].covar_module.base_kernel.raw_lengthscale.reshape(-1)

    @property
    def log_lengthscales(self) -> torch.Tensor:
        return torch.stack([self._log_ell(e) for e in range(self.n_out)], dim=1)

    @property
    def log_signal(self) -> torch.Tensor:
        return 0.5 * torch.stack([m.covar_module.raw_outputscale.reshape(())
                                  for m in self.models])

    @property
    def log_noise(self) -> torch.Tensor:
        return 0.5 * torch.stack([m.likelihood.raw_noise.reshape(())
                                  for m in self.models])

    @property
    def lengthscales(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscales)

    @property
    def signal(self) -> torch.Tensor:
        return torch.exp(self.log_signal)

    @property
    def noise(self) -> torch.Tensor:
        return torch.exp(self.log_noise)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def nlml(self) -> torch.Tensor:
        """``-log p(Y | X)`` summed over outputs.

        GPyTorch's ``ExactMarginalLogLikelihood`` returns the log likelihood
        **per data point**, hence the factor ``n``.
        """
        n = self.inputs.shape[0]
        total = torch.zeros((), dtype=self.inputs.dtype)
        for e, model in enumerate(self.models):
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            total = total - n * mll(model(self.inputs), self.targets[:, e])
        return total

    def invalidate_cache(self) -> None:
        """Drop the cached ``(A^{-1}, beta)`` (call after changing hyper-params)."""
        self._cached: Optional[tuple] = None

    def posterior(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(A_e^{-1}, beta_e = A_e^{-1} y_e)`` stacked over outputs.

        PILCO optimises the policy with the dynamics hyper-parameters held fixed,
        so this is cached for the whole inner optimisation loop.  When the
        hyper-parameters *do* require gradients (GP training) the cache is
        bypassed: a graph-attached tensor reused twice would make the second
        ``backward()`` fail.
        """
        frozen = not any(p.requires_grad for p in self.hyperparameters())
        if frozen and self._cached is not None:
            return self._cached

        n = self.inputs.shape[0]
        eye = torch.eye(n, dtype=self.inputs.dtype, device=self.inputs.device)
        inv, beta = [], []
        for e, model in enumerate(self.models):
            A = (model.covar_module(self.inputs).evaluate()
                 + model.likelihood.noise * eye)
            L = gpytorch.utils.cholesky.psd_safe_cholesky(A)
            inv.append(torch.cholesky_inverse(L))
            beta.append(torch.cholesky_solve(self.targets[:, e, None], L).squeeze(-1))
        out = (torch.stack(inv), torch.stack(beta, dim=1))
        if frozen:
            self._cached = out
        return out

    # ------------------------------------------------------------------
    # moment matching (the part GPyTorch does not provide)
    # ------------------------------------------------------------------
    def moments(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        """Predictive moments at an uncertain input ``x ~ N(mean, cov)``."""
        d, E = self.n_in, self.n_out
        dtype, device = mean.dtype, mean.device
        eye = torch.eye(d, dtype=dtype, device=device)
        iK, beta = self.posterior()

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
