"""The moment-matching operator abstraction.

Everything PILCO does analytically is an instance of one pattern:

    Given a Gaussian ``x ~ N(m, S)``, a nonlinearity ``g`` and the *linearisation
    assumption* that ``E[g(x) | x]`` is affine in ``x``, report

        E[g(x)],  Cov[g(x)],  and  C := S^{-1} Cov[x, g(x)].

The third quantity is what makes PILCO work: it lets the output be *spliced*
into a larger Gaussian, carrying the correlations with every other variable
along.  In the original MATLAB/Python code this splice is spelled out by hand
for every operator (``gTrig``, ``gSin``, ``gSat``, ``congp``, ``gp1d``, ...),
together with the Jacobians ``dMdm, dSdm, dMds, dSds, dMdp, dSdp``.

With PyTorch we do not need any of those Jacobians.  A :class:`MomentOp` only
implements the closed-form forward moments as a differentiable function of
``(m, S, parameters)``; the chain rule through the whole propagation is then
handled by autograd.  This is the central simplification of the rewrite.

The splice itself is the function :func:`splice`, a 5-line generic replacement
for the roughly 80 lines of index gymnastics in ``base/propagated.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..gaussian import symmetrize

__all__ = ["Moments", "MomentOp", "splice", "AffineOp", "as_index"]


def as_index(idx, device=None) -> torch.Tensor:
    """Normalise an index specification to a 1-D ``long`` tensor."""
    if isinstance(idx, torch.Tensor):
        return idx.long()
    if isinstance(idx, int):
        return torch.tensor([idx], dtype=torch.long)
    return torch.as_tensor(list(idx), dtype=torch.long)


@dataclass
class Moments:
    """Output moments of a :class:`MomentOp`.

    Attributes
    ----------
    mean:
        ``E[g(x)]``, shape ``(n_out,)``.
    cov:
        ``Cov[g(x)]``, shape ``(n_out, n_out)``.
    cross:
        ``S^{-1} Cov[x, g(x)]``, shape ``(n_in, n_out)``.  This is PILCO's
        ``C`` / ``V`` quantity.  It encodes how every input coordinate
        co-varies with the output and is what :func:`splice` uses to attach
        the output block to the rest of the augmented state.
    """

    mean: torch.Tensor
    cov: torch.Tensor
    cross: torch.Tensor

    @property
    def n_out(self) -> int:
        return self.mean.numel()


class MomentOp(nn.Module):
    """Base class for closed-form Gaussian moment-matching operators.

    Subclasses must implement :meth:`moments`.  The signature deliberately takes
    plain tensors (not a :class:`~pilco.gaussian.Gaussian`) so that the whole
    operator stays a pure, autograd-differentiable ``(mean, cov) -> Moments``
    map.  Parameters live in the module as usual, so ``loss.backward()``
    produces gradients w.r.t. policy parameters, GP hyper-parameters, etc.
    """

    n_in: int
    n_out: int

    def moments(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        raise NotImplementedError

    def forward(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        out = self.moments(mean, cov)
        assert out.mean.shape == (self.n_out,), (out.mean.shape, self.n_out)
        assert out.cov.shape == (self.n_out, self.n_out), (out.cov.shape, self.n_out)
        assert out.cross.shape == (self.n_in, self.n_out), (out.cross.shape, self.n_in, self.n_out)
        return out


def splice(mean: torch.Tensor, cov: torch.Tensor, op: MomentOp,
           in_idx, out_idx):
    """Attach the output of ``op`` to an augmented Gaussian.

    Parameters
    ----------
    mean, cov:
        The current augmented Gaussian, shape ``(n,)`` and ``(n, n)``.
    op:
        Operator applied to the sub-block indexed by ``in_idx``.
    in_idx:
        Coordinates of ``mean`` that are fed to ``op``.
    out_idx:
        Fresh coordinates of ``mean`` that ``op`` writes.  Must be disjoint
        from ``in_idx`` and must not yet carry information.

    Returns
    -------
    (mean, cov, moments):
        The augmented Gaussian with the output block filled in and, crucially,
        with all cross-covariances ``Cov[x_j, g(x)] = Cov[x_j, x_in] S_in^{-1}
        Cov[x_in, g]`` inserted.  Because the conditional mean is assumed
        affine, this is the only information needed.  The operator's
        :class:`Moments` are returned as well so that callers can reuse the
        ``cross`` block (for example to assemble the mean Jacobian).
    """
    in_idx = as_index(in_idx)
    out_idx = as_index(out_idx)
    if torch.isin(out_idx, in_idx).any():
        raise ValueError("in_idx and out_idx must be disjoint")

    res = op(mean[in_idx], cov[in_idx][:, in_idx])

    # Cov[x_j, g] = S[j, in] @ S_in^{-1} Cov[x_in, g] = S[j, in] @ C
    proj = cov[:, in_idx] @ res.cross                     # (n, n_out)

    new_mean = mean.index_copy(0, out_idx, res.mean)
    new_cov = cov.clone()
    new_cov = new_cov.index_copy(1, out_idx, proj)         # columns: S[:, out] = proj
    new_cov = new_cov.index_copy(0, out_idx, proj.transpose(0, 1))  # rows
    # the diagonal output block is authoritative (overwrites the two above)
    k = out_idx.numel()
    rows = out_idx.repeat_interleave(k)
    cols = out_idx.repeat(k)
    new_cov = new_cov.index_put((rows, cols), res.cov.reshape(-1))
    return new_mean, symmetrize(new_cov), res


class AffineOp(MomentOp):
    """Exact moment propagation through an affine map ``y = A x + b``.

    Provided mainly so that output/projection matrices (PILCO's ``P`` and the
    ``difi`` difference-state handling) are just another operator.
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_in = weight.shape[1]
        self.n_out = weight.shape[0]
        self.weight = nn.Parameter(weight.clone(), requires_grad=weight.requires_grad)
        if bias is None:
            bias = torch.zeros(self.n_out, dtype=weight.dtype)
        self.bias = nn.Parameter(bias.clone(), requires_grad=bias.requires_grad)

    def moments(self, mean: torch.Tensor, cov: torch.Tensor) -> Moments:
        A = self.weight
        m = A @ mean + self.bias
        S = symmetrize(A @ cov @ A.transpose(0, 1))
        # C = S_in^{-1} Cov[x, Ax+b] = S_in^{-1} S_in A^T = A^T
        return Moments(m, S, A.transpose(0, 1))
