"""Hyper-parameter training for the dynamics GP.

Replaces ``gp/train.py`` + ``gp/hypCurb.py``.  The negative log marginal
likelihood comes from GPyTorch (:meth:`~pilco.ops.gp.GpDynamicsOp.nlml`), so
training is a plain optimiser call over the GP's own parameters; the original
code needed a hand-derived NLML gradient and its own MATLAB-style ``minimize``.

Two trainers are provided, because they are not interchangeable:

``fit_hyperparameters``
    ``torch.optim.LBFGS`` with a strong-Wolfe line search, over all outputs at
    once.  Fast, but does far more work per iteration than PILCO's optimiser.

``fit_hyperparameters_pilco``
    The ported ``minimize``, one output at a time, exactly like ``gp/train.py``.
    This is what reproduces the NumPy reference, and on small datasets it is
    also what keeps the fit smooth enough for the planner to trust.

Both optimise the same object, and both apply PILCO's *curb*: a sharp
``p = 30`` penalty that pulls each ARD lengthscale towards the spread of its
input column and bounds the signal-to-noise ratio.  It is a penalty rather than
a clamp because the reference adds it to the NLML before calling ``minimize``.
"""
from __future__ import annotations

import math
from typing import Optional

import gpytorch
import torch

from ..ops.gp import GpDynamicsOp

__all__ = ["initial_log_hyp", "curbed_nlml", "fit_hyperparameters",
           "fit_hyperparameters_pilco", "build_dynamics_op",
           "LOG_ELL_MIN", "LOG_ELL_MAX", "SNR_MAX"]

LOG_ELL_MIN = math.log(1e-2)
LOG_ELL_MAX = math.log(1e2)
SNR_MAX = 1000.0


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------
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
    """Clamp a ``(d + 2, E)`` hyper-parameter tensor to PILCO's feasible set."""
    d = log_hyp.shape[0] - 2
    out = log_hyp.clone()
    out[:d] = out[:d].clamp(LOG_ELL_MIN, LOG_ELL_MAX)
    out[d + 1] = torch.minimum(out[d + 1], out[d] - math.log(SNR_MAX))
    return out


def build_dynamics_op(inputs: torch.Tensor, targets: torch.Tensor,
                      log_hyp: Optional[torch.Tensor] = None
                      ) -> GpDynamicsOp:
    """Create a :class:`GpDynamicsOp` with PILCO's initialisation."""
    if log_hyp is None:
        log_hyp = initial_log_hyp(inputs, targets)
    return GpDynamicsOp.from_log_hyp(inputs, targets, log_hyp)


# ---------------------------------------------------------------------------
# the curb penalty
# ---------------------------------------------------------------------------
def _curb_penalty(op: GpDynamicsOp, input_std: torch.Tensor,
                  ls_range: Optional[float] = None,
                  power: float = 30.0) -> torch.Tensor:
    """PILCO's ``p = 30`` penalty on the lengthscales and the SNR, all outputs.

    * ``sum(((log ell - log std(inputs)) / log 100) ** 30)`` pulls each ARD
      lengthscale towards the spread of its input column -- a strong regulariser;
    * ``sum(((log sf - log sigma_n) / log 1000) ** 30)`` bounds the
      signal-to-noise ratio.

    Omitting the first term matters in practice.  An uncurbed fit on the 40-point
    cart-pole dataset drives the noise standard deviation to ~1e-4, i.e. it fits
    the sampled process noise; the resulting overconfident model then misleads
    policy search and the closed loop fails to learn (episode cost 36/40 instead
    of ~9).  ``ls_range`` is a tighter alternative to PILCO's own ``log 100``:
    the *smoothness* of the GP, not just its marginal likelihood, decides whether
    the planner can trust it (see ``docs/ARCHITECTURE.md`` section 8).
    """
    return sum(_curb_penalty_single(op, e, input_std, ls_range, power)
               for e in range(op.n_out))


def _curb_penalty_single(op: GpDynamicsOp, e: int, input_std: torch.Tensor,
                         ls_range: Optional[float], power: float = 30.0
                         ) -> torch.Tensor:
    """The penalty for output ``e``."""
    std = torch.as_tensor(input_std, dtype=op.log_lengthscales.dtype).reshape(-1)
    if std.numel() == 1:
        std = std.expand(op.log_lengthscales.shape[0])
    scale = math.log(100.0) if ls_range is None else float(ls_range)
    ll = op.log_lengthscales[:, e] - torch.log(std.clamp(min=1e-6))
    snr = (op.log_signal[e] - op.log_noise[e]) / math.log(SNR_MAX)
    return (ll / scale).pow(power).sum() + snr.pow(power)


def curbed_nlml(op: GpDynamicsOp, input_std: torch.Tensor,
                ls_range: Optional[float] = None) -> torch.Tensor:
    """``nlml + curb`` -- what PILCO actually minimises over the hyper-parameters."""
    return op.nlml() + _curb_penalty(op, input_std, ls_range)


# ---------------------------------------------------------------------------
# trainer 1: all outputs at once, torch LBFGS
# ---------------------------------------------------------------------------
def fit_hyperparameters(op: GpDynamicsOp, iterations: int = 300,
                        tolerance: float = 1e-8,
                        verbose: bool = False,
                        use_curb: bool = True,
                        noise_floor: float = 0.0,
                        ls_range: Optional[float] = None) -> GpDynamicsOp:
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
        :func:`_curb_penalty`).  Disabling it is useful for experiments but lets
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
    params = op.hyperparameters()
    input_std = torch.std(op.inputs, dim=0)

    def project():
        """Keep the hyper-parameters on PILCO's feasible set.

        Writing through every model is idempotent, so the shared-lengthscale case
        needs no special handling.
        """
        with torch.no_grad():
            for model in op.models:
                model.covar_module.base_kernel.raw_lengthscale.clamp_(
                    LOG_ELL_MIN, LOG_ELL_MAX)
                model.likelihood.raw_noise.copy_(torch.minimum(
                    model.likelihood.raw_noise,
                    model.covar_module.raw_outputscale - 2.0 * math.log(SNR_MAX)))
                if noise_floor > 0.0:
                    model.likelihood.raw_noise.clamp_(min=2.0 * math.log(noise_floor))

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
        loss = (curbed_nlml(op, input_std, ls_range=ls_range)
                if use_curb else op.nlml())
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


# ---------------------------------------------------------------------------
# trainer 2: one output at a time, PILCO's own minimise
# ---------------------------------------------------------------------------
def fit_hyperparameters_pilco(op: GpDynamicsOp, iterations: int = 150,
                              MFEPLS: int = 30, power: float = 30.0,
                              ls_range: Optional[float] = None,
                              verbose: bool = False) -> GpDynamicsOp:
    """Train the GP with **PILCO's own optimiser**, one output at a time.

    ``gp/train.py`` loops over outputs and calls ``minimize`` separately on each
    column of ``hyp``; this reproduces that exactly, using the PyTorch port in
    :mod:`pilco.optim.minimize`.  Using the reference optimiser is what makes the
    GP fit land in the reference's basin (see
    ``scripts/diagnostics/reproduction_check.py``).

    The objective is GPyTorch's marginal likelihood (:meth:`GpDynamicsOp.nlml`
    restricted to one output) plus the curb.  Because the per-output GP's raw
    parameters *are* the log hyper-parameters, ``minimize`` sees exactly the
    coordinates it saw in the reference implementation.
    """
    from ..optim.minimize import ParamVector, autograd_objective, minimize

    input_std = torch.std(op.inputs, dim=0)
    n = op.inputs.shape[0]
    for e, model in enumerate(op.models):
        params = [model.covar_module.base_kernel.raw_lengthscale,
                  model.covar_module.raw_outputscale,
                  model.likelihood.raw_noise]
        vec = ParamVector(params)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        def objective(model=model, mll=mll, e=e):
            # GPyTorch's MLL is per data point, hence the factor ``n``.
            return (-n * mll(model(op.inputs), op.targets[:, e])
                    + _curb_penalty_single(op, e, input_std, ls_range, power))

        # seed the optimiser from the current hyper-parameters, exactly like
        # PILCO (``lh[:, i]`` is warm-started across episodes)
        with torch.no_grad():
            params[0].clamp_(LOG_ELL_MIN, LOG_ELL_MAX)
        res = minimize(vec.flat(), autograd_objective(objective, vec),
                       length=iterations, MFEPLS=MFEPLS)
        vec.set(res.x)
        op.invalidate_cache()
        if verbose:
            print(f"  GP {e + 1}/{op.n_out}: {len(res.fX)} line searches, "
                  f"NLML {res.fX[0]:.6e} -> {res.final:.6e}")
    return op
