"""Saturating cost as a closed-form moment operator.

The cost implemented here is PILCO's ``loss/lossSat.py``::

    c(x) = 1 - exp(-1/2 (x - z)^T W (x - z)),          x ~ N(m, S)

Optionally averaged over a scale mixture of widths (``cost['width']`` in the
cart-pole scenario).  Writing ``d = m - z`` and ``A = I + S W``::

    e1  = E[exp(-1/2 d^T W d)] = |A|^{-1/2} exp(-1/2 d^T A^{-1} W d)
    e2  = E[exp(-    d^T W d)] = |I + 2 S W|^{-1/2} exp(-d^T (I + 2 S W)^{-1} W d)
    E[c]    = 1 - e1
    Var[c]  = e2 - e1^2

For two jointly Gaussian states the *cross* moment uses the same formula on the
stacked vector with block-diagonal ``W``::

    E[c1 c2] = 1 - E[c1] - E[c2] + E[exp(-1/2 d12^T W2 d12)]

where ``d12 = [m1-z; m2-z]`` and ``W2 = blkdiag(W, W)``.  This is exactly the
construction ``base/analytical_variance.py`` needs for
``Cov(cost(X_t), cost(X_t'))``.

The original code also hand-derives ``dLdm, dLds, dSdm, dSds``.  Here autograd
provides them, so the operator is a two-line differentiable expression.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..gaussian import logdet, symmetrize

__all__ = ["SaturatingCostOp"]


class SaturatingCostOp(nn.Module):
    """``1 - exp(-1/2 (x - target)^T W (x - target))`` with ``W = Q / width^2``.

    Parameters
    ----------
    target:
        ``(n,)`` target state ``z``.
    weight:
        ``(n, n)`` base weight matrix ``Q``.
    widths:
        Scale mixture widths; the cost is the average over
        ``W_k = Q / width_k^2``.
    """

    def __init__(self, target: torch.Tensor, weight: torch.Tensor,
                 widths: Sequence[float] = (1.0,)):
        super().__init__()
        target = torch.as_tensor(target).reshape(-1)
        weight = torch.as_tensor(weight)
        if weight.shape != (target.numel(), target.numel()):
            raise ValueError("weight must be (n, n) matching target")
        self.n = target.numel()
        self.register_buffer("target", target.clone())
        self.register_buffer("weight", symmetrize(weight.clone()))
        widths = torch.as_tensor(list(widths), dtype=target.dtype).reshape(-1)
        self.register_buffer("widths", widths)

    # ------------------------------------------------------------------
    def _e1(self, mean: torch.Tensor, cov: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        d = mean - self.target
        A = torch.eye(self.n, dtype=mean.dtype, device=mean.device) + symmetrize(cov) @ W
        L = torch.linalg.solve(A.transpose(0, 1), (W @ d).unsqueeze(-1)).squeeze(-1)
        return torch.exp(-0.5 * logdet(A) - 0.5 * (d * L).sum())

    def _e2(self, mean: torch.Tensor, cov: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        d = mean - self.target
        B = torch.eye(self.n, dtype=mean.dtype, device=mean.device) + 2.0 * symmetrize(cov) @ W
        L = torch.linalg.solve(B.transpose(0, 1), (W @ d).unsqueeze(-1)).squeeze(-1)
        return torch.exp(-0.5 * logdet(B) - (d * L).sum())

    # ------------------------------------------------------------------
    def moments(self, mean: torch.Tensor, cov: torch.Tensor):
        """Return ``(E[c], Var[c])``."""
        mean = torch.as_tensor(mean)
        cov = symmetrize(torch.as_tensor(cov))
        e1 = torch.zeros((), dtype=mean.dtype, device=mean.device)
        e2 = torch.zeros((), dtype=mean.dtype, device=mean.device)
        for w in self.widths:
            W = self.weight / (w ** 2)
            e1 = e1 + self._e1(mean, cov, W)
            e2 = e2 + self._e2(mean, cov, W)
        k = self.widths.numel()
        e1, e2 = e1 / k, e2 / k
        value = 1.0 - e1
        variance = torch.clamp(e2 - e1 ** 2, min=0.0)
        return value, variance

    def expected(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        return self.moments(mean, cov)[0]

    # ------------------------------------------------------------------
    def cross_moment(self, mean1: torch.Tensor, mean2: torch.Tensor,
                     cov1: torch.Tensor, cov2: torch.Tensor,
                     cov12: torch.Tensor) -> torch.Tensor:
        """``E[c(x1) c(x2)]`` for a jointly Gaussian ``(x1, x2)``.

        Writing ``c = 1 - e^{-a}`` with ``a = 1/2 d^T W d``::

            E[c1 c2] = 1 - E[e^{-a1}] - E[e^{-a2}] + E[e^{-a1-a2}]

        so the *exponential* expectations are needed, not the cost expectations.
        (Getting this wrong flips the sign of the whole cross term; the NumPy
        reference's ``cost_cross_covariance`` uses the equivalent, correct form
        ``(1 - m_c) - (1 - m_1)(1 - m_2)`` with ``m`` the joint/individual cost
        means.)  The last term is again a Gaussian integral -- the same
        ``_e1``-style expression on the stacked system with block-diagonal ``W``.
        """
        n = self.n
        dev = mean1.device
        dtype = mean1.dtype
        exp1 = torch.zeros((), dtype=dtype, device=dev)      # E[e^{-a1}]
        exp2 = torch.zeros((), dtype=dtype, device=dev)
        e_joint = torch.zeros((), dtype=dtype, device=dev)
        stacked_cov = symmetrize(torch.cat([
            torch.cat([cov1, cov12], dim=1),
            torch.cat([cov12.transpose(0, 1), cov2], dim=1),
        ], dim=0))
        stacked_mean = torch.cat([mean1, mean2])
        stacked_target = torch.cat([self.target, self.target])
        for w in self.widths:
            W = self.weight / (w ** 2)
            exp1 = exp1 + self._e1(mean1, cov1, W)
            exp2 = exp2 + self._e1(mean2, cov2, W)
            W2 = torch.zeros(2 * n, 2 * n, dtype=dtype, device=dev)
            W2[:n, :n] = W
            W2[n:, n:] = W
            d = stacked_mean - stacked_target
            A = torch.eye(2 * n, dtype=dtype, device=dev) + stacked_cov @ W2
            L = torch.linalg.solve(A.transpose(0, 1),
                                   (W2 @ d).unsqueeze(-1)).squeeze(-1)
            e_joint = e_joint + torch.exp(-0.5 * logdet(A) - 0.5 * (d * L).sum())
        k = self.widths.numel()
        return 1.0 - (exp1 + exp2) / k + e_joint / k


class TrigAugmentedCostOp(nn.Module):
    """PILCO's ``loss_cp``-style cost: augment the state with sin/cos, then
    apply :class:`SaturatingCostOp` on the augmented vector.

    This wraps the two steps so that the planning code can keep treating the
    cost as ``(state_mean, state_cov) -> (value, variance)`` -- exactly the
    interface ``loss/loss_cp.py`` offers in the NumPy reference, where the
    trigonometric augmentation is done inside the loss.

    Parameters
    ----------
    angle_indices:
        Coordinates of the state that are angles.
    state_dim:
        Dimension of the physical state.
    target / weight / widths:
        As for :class:`SaturatingCostOp`, but expressed on the *augmented*
        vector ``[state | sin(angles) | cos(angles)]``.
    """

    def __init__(self, angle_indices, state_dim: int, target: torch.Tensor,
                 weight: torch.Tensor, widths=(1.0,)):
        super().__init__()
        from .fourier import TrigAugmentOp

        self.angle_indices = tuple(int(a) for a in angle_indices)
        self.state_dim = int(state_dim)
        self.trig = TrigAugmentOp(self.angle_indices, n_in=self.state_dim)
        self.cost = SaturatingCostOp(target, weight, widths)

    # ------------------------------------------------------------------
    def augment(self, mean: torch.Tensor, cov: torch.Tensor):
        """Return ``(aug_mean, aug_cov)`` for ``[state | sin | cos]``."""
        t = self.trig.moments(mean, cov)
        d, A = self.state_dim, len(self.angle_indices)
        # the trig operator consumes the *whole* state, so its C is (d, 2A)
        proj = cov @ t.cross
        aug_cov = torch.zeros(d + 2 * A, d + 2 * A, dtype=cov.dtype,
                              device=cov.device)
        aug_cov[:d, :d] = cov
        aug_cov[:d, d:] = proj
        aug_cov[d:, :d] = proj.transpose(0, 1)
        aug_cov[d:, d:] = t.cov
        return torch.cat([mean, t.mean]), aug_cov

    def moments(self, mean: torch.Tensor, cov: torch.Tensor):
        aug_m, aug_S = self.augment(mean, cov)
        return self.cost.moments(aug_m, aug_S)

    def expected(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        return self.moments(mean, cov)[0]

    def cross_moment(self, mean1, mean2, cov1, cov2, cov12) -> torch.Tensor:
        """``E[c(x1) c(x2)]`` with the trig augmentation handled jointly."""
        from .joint import augmented_joint

        m1, m2, s1, s2, s12 = augmented_joint(
            mean1, cov1, mean2, cov2, cov12, self.angle_indices)
        return self.cost.cross_moment(m1, m2, s1, s2, s12)
