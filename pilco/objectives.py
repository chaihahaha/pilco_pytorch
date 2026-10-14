"""Objectives for PILCO and for directed exploration.

Four differentiable objectives live here, all built from the closed-form moment
propagation in :mod:`pilco.trajectory`:

``expected_cumulative_cost``
    PILCO's original objective ``J(psi) = sum_t gamma^{t+1} E[c(x_t)]``.

``cumulative_cost_variance``
    The *correct* analytic ``Var[sum_t gamma^{t+1} c(x_t)]`` including the
    cross-timestep covariance terms that Deisenroth's thesis reports as
    contributing 40-85% of the total (thesis eq. 4.24-4.36).  The cross
    covariances are built from the forward-mode Jacobians of the one-step map,
    which autograd provides exactly:
    ``Cov(X_a, X_b) = S_a J_a^T J_{a+1}^T ... J_{b-1}^T``.

``information_gain``
    A closed-form, differentiable *model-improvement* objective.  For a GP with
    kernel ``k_e`` and noise ``sigma_e`` the mutual information between the
    observations at the query points ``z_0..z_{H-1}`` and the latent function is
    ``0.5 log|I + diag(sigma_e^{-2}) Kbar_e|``, where ``Kbar_e[t,s] =
    E[k_e(z_t, z_s)]`` is the *expected* kernel matrix under PILCO's Gaussian
    trajectory distribution.  That expectation has a closed form for the
    squared-exponential kernel, so the whole quantity is a differentiable
    function of the policy parameters -- this is the "closed-form directed
    exploration" objective, as opposed to the Monte-Carlo candidate search used
    in ``pilco_python``.  It directly rewards trajectories whose query points are
    *informative* (low redundancy in the GP's own metric) rather than merely
    uncertain.

``acquisition``
    UCB / Gittins-index / PI / EI wrappers around ``(mean, variance)`` of the
    cumulative cost, matching ``base/directed_explore.py`` but with a variance
    that now includes the cross terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .gaussian import symmetrize
from .ops.base import MomentOp
from .trajectory import PilcoModel, StateSpec, Trajectory

__all__ = [
    "expected_cumulative_cost", "cumulative_cost_variance",
    "cost_cross_covariances", "information_gain", "expected_kernel_matrix",
    "ucb", "gittins_index", "solve_gittins_lambda_prime",
    "expected_improvement", "probability_of_improvement",
    "ExplorationObjective",
]


# ---------------------------------------------------------------------------
# PILCO's original objective
# ---------------------------------------------------------------------------
def expected_cumulative_cost(traj: Trajectory) -> torch.Tensor:
    """``sum_t gamma^{t+1} E[c(x_t)]`` -- PILCO's exploitation objective."""
    return traj.value()


# ---------------------------------------------------------------------------
# cumulative cost distribution
# ---------------------------------------------------------------------------
def cost_cross_covariances(traj: Trajectory) -> dict:
    """Covariances ``Cov(X_a, X_b)`` of the *cost* states.

    The cost is evaluated at ``traj.states[1] .. traj.states[H]`` (the state
    *after* each transition), so the returned matrices are indexed by cost step
    ``t in 0..H-1`` and refer to ``traj.states[t + 1]``.

    The recursion is the linear-Gaussian chain
    ``Cov(X_a, X_b) = Cov(X_a, X_{b-1}) J_{b-1}^T`` with
    ``J_i = d mean_{i+1} / d mean_i``.  Note the transposes: the NumPy reference
    ``base/analytical_variance.py::compute_state_cross_covariances`` uses
    ``S_a J_a J_{a+1}...`` (no transposes, wrong product order) and the audit
    measured relative errors of 0.89-2.4 against the exact answer.
    """
    H = traj.horizon
    J = traj.jacobians
    cov = [[None] * H for _ in range(H)]
    for t in range(H):
        cov[t][t] = symmetrize(traj.states[t + 1].cov)
    for tau in range(1, H):
        for t in range(H - tau):
            cov[t][t + tau] = cov[t][t + tau - 1] @ J[t + tau].transpose(0, 1)
    return cov


def cumulative_cost_variance(traj: Trajectory, model: PilcoModel,
                             include_cross_terms: bool = True) -> torch.Tensor:
    """``Var[sum_t gamma^{t+1} c(x_t)]``.

    ``include_cross_terms=False`` reproduces the (incorrect) sum-of-marginal
    variances proxy used by ``base/value_de.py::_compute_marginal_variance``;
    with cross terms the value is the analytically correct one.
    """
    gamma = traj.gamma
    H = traj.horizon
    disc = torch.as_tensor([gamma ** (t + 1) for t in range(H)],
                           dtype=traj.costs.dtype, device=traj.costs.device)
    total = (disc ** 2 * traj.cost_vars).sum()
    if not include_cross_terms or H < 2:
        return torch.clamp(total, min=1e-12)
    cov = cost_cross_covariances(traj)
    for t in range(H):
        for s in range(t + 1, H):
            cross_moment = model.cost.cross_moment(
                traj.states[t + 1].mean, traj.states[s + 1].mean,
                traj.states[t + 1].cov, traj.states[s + 1].cov, cov[t][s])
            cov_cs = cross_moment - traj.costs[t] * traj.costs[s]
            total = total + 2.0 * disc[t] * disc[s] * cov_cs
    return torch.clamp(total, min=1e-12)


# ---------------------------------------------------------------------------
# closed-form information gain
# ---------------------------------------------------------------------------
def expected_kernel_matrix(dynamics, means: torch.Tensor, covs: torch.Tensor,
                           e: int, cross_covs=None):
    """``Kbar[t,s] = E[k_e(z_t, z_s)]`` for ``z_t ~ N(means[t], covs[t])``.

    For the squared-exponential kernel the expectation is closed form::

        E[k_e(z_t,z_s)] = sf_e^2 |I + L V|^{-1/2}
                          exp(-1/2 delta^T L (I + V L)^{-1} delta)

    with ``delta = mu_t - mu_s``, ``L = diag(1/ell_e^2)`` and
    ``V = Var(z_t - z_s)``.  When the query points are treated as independent,
    ``V = covs[t] + covs[s]``; passing ``cross_covs[t][s] = Cov(z_t, z_s)``
    uses the exact ``V = covs[t] + covs[s] - C - C^T``.  The diagonal ``t = s``
    is exact either way and equals ``sf_e^2``.
    """
    dtype = means.dtype
    lam = torch.exp(-2.0 * dynamics.log_lengthscales[:, e])       # 1/ell_e^2
    n = means.shape[0]
    eye = torch.eye(lam.numel(), dtype=dtype, device=means.device)
    Kbar = torch.zeros(n, n, dtype=dtype, device=means.device)
    for t in range(n):
        for s in range(n):
            if t == s:
                Kbar[t, s] = dynamics.signal[e] ** 2
                continue
            delta = means[t] - means[s]
            V = covs[t] + covs[s]
            if cross_covs is not None:
                C = cross_covs[t][s]
                V = V - C - C.transpose(0, 1)
            V = symmetrize(V)
            B = eye + (lam[:, None] * lam[None, :]) * V
            q = -0.5 * (delta * torch.linalg.solve(B, (lam * delta))).sum()
            Kbar[t, s] = dynamics.signal[e] ** 2 * torch.exp(
                q - 0.5 * torch.logdet(B))
    return Kbar


def information_gain(traj: Trajectory, model: PilcoModel,
                     spec: Optional[StateSpec] = None,
                     normalized: bool = True) -> torch.Tensor:
    """Closed-form GP mutual information of the planned query points.

    ``I(y; f) = 0.5 sum_e log det(I + sigma_e^{-2} Kbar_e)`` where ``Kbar_e`` is
    the *expected* kernel matrix over the planned GP inputs (see
    :func:`expected_kernel_matrix`).  This is the GP-UCB / D-optimal design
    criterion (Srinivas et al. 2010) evaluated in closed form under PILCO's
    moment-matched trajectory.

    Because the diagonal of ``Kbar_e`` is exactly ``sf_e^2`` -- independent of
    the policy -- the raw criterion is dominated by the constant
    ``0.5 * n * log(1 + sf_e^2/sigma_e^2)``.  With ``normalized=True`` that
    constant is subtracted and the result divided by the number of query
    points, giving a quantity in ``[0, log(1+sf^2/sigma^2))`` that measures how
    *non-redundant* the visited inputs are: 0 means the policy keeps revisiting
    the same point, the maximum means every query point is independent.

    Caveats worth knowing before using this as an exploration objective:

    * the off-diagonal ``E[k(z_t, z_s)]`` currently treats ``z_t`` and ``z_s``
      as independent; using the exact joint would *increase* the redundancy and
      so lower the score.  Pass an explicit ``cross_covs`` to
      :func:`expected_kernel_matrix` to do so.
    * the criterion is *undirected* with respect to cost: it rewards covering
      distinct inputs, not distinct *relevant* inputs.  Combining it with the
      cost term is what keeps it on-task.
    """
    spec = spec or model.spec
    means, covs = traj.gp_inputs(spec)                    # (H, m), (H, m, m)
    n = means.shape[0]
    if n < 2:
        return torch.zeros((), dtype=means.dtype, device=means.device)
    total = torch.zeros((), dtype=means.dtype, device=means.device)
    eye = torch.eye(n, dtype=means.dtype, device=means.device)
    for e in range(model.dynamics.n_out):
        Kbar = expected_kernel_matrix(model.dynamics, means, covs, e)
        noise_var = model.dynamics.noise[e] ** 2
        diag = model.dynamics.signal[e] ** 2
        total = total + 0.5 * torch.logdet(eye + Kbar / noise_var)
        if normalized:
            total = total - 0.5 * n * torch.log1p(diag / noise_var)
    if normalized:
        total = total / (n * model.dynamics.n_out)
    return total


# ---------------------------------------------------------------------------
# Bayesian-optimisation acquisition functions
# ---------------------------------------------------------------------------
def ucb(value: torch.Tensor, variance: torch.Tensor,
        beta: float = 1.0) -> torch.Tensor:
    """Upper confidence bound on the cost: ``mu - beta * sigma`` (minimised)."""
    return value - beta * torch.sqrt(torch.clamp(variance, min=1e-12))


def solve_gittins_lambda_prime(gamma: float, max_iter: int = 100,
                               tol: float = 1e-12) -> float:
    """Solve ``gamma (phi(l) + l Phi(l) - l) + l = 0`` for ``l <= 0``.

    This is the Gittins index slope of Deisenroth's thesis, Table 4.1.  The
    solution is independent of the controller, hence the acquisition value
    ``mu + lambda' * sigma`` is differentiable with a constant coefficient.
    """
    from math import erf, exp, pi, sqrt

    def phi(x):
        return exp(-0.5 * x * x) / sqrt(2 * pi)

    def Phi(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    lam = 0.0
    for _ in range(max_iter):
        g = gamma * (phi(lam) + lam * Phi(lam) - lam) + lam
        dg = gamma * (Phi(lam) - 1.0) + 1.0
        if abs(g) < tol:
            break
        step = g / dg
        lam = lam - step
    return lam


def gittins_index(value: torch.Tensor, variance: torch.Tensor,
                  episodes_remaining: int) -> torch.Tensor:
    """Gittins-index acquisition ``mu + lambda'_gamma * sigma`` (minimised)."""
    if episodes_remaining <= 1:
        return value
    gamma = 1.0 - 1.0 / episodes_remaining
    if gamma <= 0.0:
        return value
    lam = solve_gittins_lambda_prime(gamma)
    return value + lam * torch.sqrt(torch.clamp(variance, min=1e-12))


def expected_improvement(value: torch.Tensor, variance: torch.Tensor,
                         best_value: float, best_variance: float = 0.0):
    """``EI`` for a *cost* to be minimised (returned negated for minimisation)."""
    from math import sqrt

    s = torch.sqrt(torch.clamp(variance + best_variance, min=1e-12))
    z = (best_value - value) / s
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=value.dtype), torch.ones((), dtype=value.dtype))
    Phi = normal.cdf(z)
    phi = torch.exp(-0.5 * z * z) / sqrt(2.0 * 3.141592653589793)
    return (value - best_value) * Phi + s * phi


def probability_of_improvement(value: torch.Tensor, variance: torch.Tensor,
                               best_value: float, best_variance: float = 0.0):
    """``-P(cost < best)`` (returned negated for minimisation)."""
    s = torch.sqrt(torch.clamp(variance + best_variance, min=1e-12))
    z = (best_value - value) / s
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=value.dtype), torch.ones((), dtype=value.dtype))
    return -normal.cdf(z)


# ---------------------------------------------------------------------------
# combination
# ---------------------------------------------------------------------------
@dataclass
class ExplorationObjective:
    """How to score a candidate policy.

    ``kind`` selects the exploitation term:

    * ``"mean"``    -- plain PILCO, ``E[C]``
    * ``"ucb"``     -- ``E[C] - beta * sd[C]``
    * ``"gittins"`` -- ``E[C] + lambda'_gamma * sd[C]``
    * ``"base"``    -- plain PILCO

    ``info_weight`` adds ``- info_weight * information_gain``, the closed-form
    model-improvement term.  ``cross_terms`` switches the correct analytic
    variance (with cross-timestep covariances) on; disabling it reproduces the
    sum-of-marginal-variance proxy used by the NumPy implementation.
    """

    kind: str = "mean"
    beta: float = 1.0
    info_weight: float = 0.0
    cross_terms: bool = True
    episodes_remaining: int = 10

    def __call__(self, traj: Trajectory, model: PilcoModel) -> torch.Tensor:
        value = expected_cumulative_cost(traj)
        if self.kind == "mean" and self.info_weight == 0.0:
            return value
        variance = cumulative_cost_variance(traj, model,
                                            include_cross_terms=self.cross_terms)
        if self.kind == "ucb":
            score = ucb(value, variance, self.beta)
        elif self.kind == "gittins":
            score = gittins_index(value, variance, self.episodes_remaining)
        elif self.kind == "mean":
            score = value
        else:
            raise ValueError(f"unknown objective kind {self.kind!r}")
        if self.info_weight:
            score = score - self.info_weight * information_gain(traj, model)
        return score
