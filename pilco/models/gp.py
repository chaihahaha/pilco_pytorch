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


def curbed_nlml(op: GpDynamicsOp, input_std: torch.Tensor,
                ls_max: float = 100.0, snr_max: float = SNR_MAX,
                power: float = 30.0) -> torch.Tensor:
    """PILCO's curbed negative log marginal likelihood (``gp/hypCurb.py``).

    The original does not *clamp* the hyper-parameters, it adds a sharp
    (``p = 30``) **penalty** to the NLML:

    * ``sum(((ll - log std(inputs)) / log 100) ** 30)`` pulls each ARD
      lengthscale towards the standard deviation of the corresponding input
      column -- a strong regulariser;
    * ``sum(((log sf - log sigma_n) / log 1000) ** 30)`` bounds the
      signal-to-noise ratio.

    Omitting the first term matters a lot in practice.  An uncurbed L-BFGS on
    the same 40-point cart-pole dataset used by the NumPy reference drives the
    noise standard deviation to ~1e-4 and the NLML *below* the reference's
    optimum, i.e. it fits the sampled process noise; the resulting overconfident
    model then misleads policy search and the closed loop fails to learn
    cart-pole (episode cost 36/40 instead of ~9).  Reproducing the penalty makes
    the two trainers agree and the loop learn.
    """
    f = op.nlml()
    std = torch.as_tensor(input_std, dtype=op.log_lengthscales.dtype).reshape(-1)
    if std.numel() == 1:
        std = std.expand(op.log_lengthscales.shape[0])
    ll = op.log_lengthscales
    f = f + (((ll - torch.log(std.clamp(min=1e-6))[:, None])
              / math.log(ls_max)) ** power).sum()
    f = f + (((op.log_signal - op.log_noise) / math.log(snr_max)) ** power).sum()
    return f


def fit_hyperparameters(op: GpDynamicsOp, iterations: int = 300,
                        tolerance: float = 1e-8,
                        verbose: bool = False,
                        use_curb: bool = True,
                        noise_floor: float = 0.0) -> GpDynamicsOp:
    """Maximise the curbed marginal likelihood with L-BFGS (in place).

    Parameters
    ----------
    op:
        The GP operator whose hyper-parameters are optimised.  Training inputs
        and targets are the buffers already held by the operator.
    iterations:
        Maximum number of L-BFGS iterations.
    use_curb:
        Apply PILCO's sharp penalty on the lengthscales and the SNR (see
        :func:`curbed_nlml`).  Disabling it is useful for experiments but lets
        the fit run away on small datasets.
    noise_floor:
        Lower bound on the observation-noise standard deviation.  Motivation:
        the GP targets are *state differences* from a rollout that already
        contains process noise, so their observation noise has a physical floor,
        while an unconstrained marginal-likelihood maximiser drives it to ~1e-4.

        **Measured on cart-pole, this does not help and is off by default.**
        ``scripts/diagnostics/gp_fit_effect.py`` reports the policy value after
        the same optimiser for: NumPy's fit 33.85, torch's unconstrained fit
        35.80, floor 0.01 -> 39.14, floor 0.005 -> 39.33, floor 0.02 -> 35.25.
        The likelihood wants the noise small (constraining it raises the NLML
        from -17.6 to 1739 at floor 0.01), so the floor mostly distorts the
        lengthscales.  It is kept as an option because it is the right knob for a
        genuinely noisy plant, but it is not the fix for the closed-loop gap.
    """
    params = [op.log_lengthscales, op.log_signal, op.log_noise]
    input_std = torch.std(op.inputs, dim=0)

    def project():
        """Keep the hyper-parameters on PILCO's feasible set."""
        with torch.no_grad():
            op.log_lengthscales.clamp_(LOG_ELL_MIN, LOG_ELL_MAX)
            op.log_noise.copy_(torch.minimum(
                op.log_noise, op.log_signal - math.log(SNR_MAX)))
            if noise_floor > 0.0:
                op.log_noise.clamp_(min=math.log(noise_floor))

    project()
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
        project()
        loss = (curbed_nlml(op, input_std) if use_curb else op.nlml())
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
