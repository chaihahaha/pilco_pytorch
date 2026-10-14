"""Hyper-parameter training for the dynamics GP.

Replaces ``gp/train.py`` + ``gp/hypCurb.py``.  Because
:class:`~pilco.ops.gp.GpDynamicsOp` exposes its negative log marginal likelihood
as a differentiable function of the log-hyper-parameters, training is just a
``torch.optim.LBFGS`` call.  The original code needed a hand-derived gradient of
the NLML and a separate MATLAB-style ``minimize`` with a custom line search.

PILCO's "hyper-parameter curb" (bound the signal-to-noise ratio and the
lengthscales) is implemented as a projection applied inside the LBFGS closure,
which keeps the optimiser on the feasible set without adding penalty terms.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from ..ops.gp import GpDynamicsOp

__all__ = ["initial_log_hyp", "fit_hyperparameters", "build_dynamics_op"]

LOG_ELL_MIN = math.log(1e-2)
LOG_ELL_MAX = math.log(1e2)
SNR_MAX = 1000.0


def initial_log_hyp(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """PILCO's heuristic initialisation of ``hyp`` (see ``gp/train.py``).

    ``log lengthscales = log std(inputs)``, ``log signal std = log std(targets)``,
    ``log noise std = log(std(targets) / 10)``.
    """
    d = inputs.shape[1]
    E = targets.shape[1]
    std_in = torch.std(inputs, dim=0)
    std_out = torch.std(targets, dim=0).clamp(min=1e-6)
    log_hyp = torch.zeros(d + 2, E, dtype=inputs.dtype, device=inputs.device)
    log_hyp[:d] = torch.log(std_in.clamp(min=1e-3)).unsqueeze(1).expand(d, E)
    log_hyp[d] = torch.log(std_out)
    log_hyp[d + 1] = torch.log(std_out / 10.0)
    return _curb(log_hyp)


def _curb(log_hyp: torch.Tensor) -> torch.Tensor:
    """PILCO's hyper-parameter constraints (lengthscales and SNR)."""
    d = log_hyp.shape[0] - 2
    out = log_hyp.clone()
    out[:d] = out[:d].clamp(LOG_ELL_MIN, LOG_ELL_MAX)
    # log noise std <= log signal std - log(SNR_MAX)
    out[d + 1] = torch.minimum(out[d + 1], out[d] - math.log(SNR_MAX))
    return out


def build_dynamics_op(inputs: torch.Tensor, targets: torch.Tensor,
                      log_hyp: Optional[torch.Tensor] = None
                      ) -> GpDynamicsOp:
    """Create a :class:`GpDynamicsOp` with PILCO's initialisation."""
    if log_hyp is None:
        log_hyp = initial_log_hyp(inputs, targets)
    return GpDynamicsOp.from_log_hyp(inputs, targets, log_hyp)


def fit_hyperparameters(op: GpDynamicsOp, iterations: int = 300,
                        tolerance: float = 1e-8,
                        verbose: bool = False) -> GpDynamicsOp:
    """Maximise the marginal likelihood with L-BFGS (in place).

    Parameters
    ----------
    op:
        The GP operator whose hyper-parameters are optimised.  Training inputs
        and targets are the buffers already held by the operator.
    iterations:
        Maximum number of L-BFGS iterations (PILCO uses 300-500 line searches).
    """
    params = [op.log_lengthscales, op.log_signal, op.log_noise]
    with torch.no_grad():
        op.log_lengthscales.clamp_(LOG_ELL_MIN, LOG_ELL_MAX)
        op.log_noise.copy_(torch.minimum(op.log_noise,
                                         op.log_signal - math.log(SNR_MAX)))
    optimiser = torch.optim.LBFGS(params, lr=1.0, max_iter=iterations,
                                  tolerance_grad=tolerance,
                                  tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    history: list[float] = []

    def closure():
        optimiser.zero_grad()
        # Project *before* evaluating: L-BFGS probes points outside the feasible
        # set and an unbounded lengthscale can make the Gram matrix numerically
        # singular.
        with torch.no_grad():
            op.log_lengthscales.clamp_(LOG_ELL_MIN, LOG_ELL_MAX)
            op.log_noise.copy_(torch.minimum(
                op.log_noise, op.log_signal - math.log(SNR_MAX)))
        loss = op.nlml()
        loss.backward()
        for p in params:
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()
        history.append(float(loss.detach()))
        return loss

    optimiser.step(closure)
    op.invalidate_cache()
    if verbose:
        print(f"  GP NLML {history[0]:.4e} -> {history[-1]:.4e} "
              f"({len(history)} evals)")
    return op
