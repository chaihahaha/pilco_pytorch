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

from ..ops.gp import GpDynamicsOp, _robust_cholesky

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
                power: float = 30.0,
                ls_range: Optional[float] = None) -> torch.Tensor:
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
    # ``ls_max`` reproduces PILCO's own curb; ``ls_range`` (in log units) is a
    # tighter alternative that keeps the lengthscale within ``exp(ls_range)`` of
    # the input spread.  It matters because the *smoothness* of the GP, not just
    # its marginal likelihood, decides whether the planner can trust it: on the
    # 40-point cart-pole dataset the reference's fit sits at log-lengthscales
    # [5.15, 2.06, 5.95, 2.92, 3.68] (a very smooth model) while an unconstrained
    # fit goes to [2.73, 1.15, 2.54, 0.002, 2.72], and the smooth model yields a
    # 6% better policy value.
    scale = (math.log(ls_max) if ls_range is None else float(ls_range))
    f = f + (((ll - torch.log(std.clamp(min=1e-6))[:, None]) / scale) ** power).sum()
    f = f + (((op.log_signal - op.log_noise) / math.log(snr_max)) ** power).sum()
    return f


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
# Per-output training with PILCO's own optimiser
# ---------------------------------------------------------------------------
def gp_nlml(inputs: torch.Tensor, targets: torch.Tensor,
            log_lengthscales: torch.Tensor, log_signal: torch.Tensor,
            log_noise: torch.Tensor,
            individual_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Negative log marginal likelihood for a *single* output.

    Equivalent to ``GpDynamicsOp.nlml()`` restricted to one column, but written
    as a free function of the hyper-parameters so PILCO's per-output optimisation
    (``gp/train.py`` loops over outputs and calls ``minimize`` separately) can be
    reproduced exactly.
    """
    iell = torch.exp(-log_lengthscales)                     # (d,)
    diff = (inputs[:, None, :] - inputs[None, :, :]) * iell
    K = torch.exp(2.0 * log_signal) * torch.exp(-0.5 * (diff ** 2).sum(-1))
    n = inputs.shape[0]
    A = K + torch.exp(2.0 * log_noise) * torch.eye(n, dtype=K.dtype,
                                                   device=K.device)
    if individual_noise is not None:
        A = A + torch.diag(individual_noise)
    L = _robust_cholesky(A)
    alpha = torch.cholesky_solve(targets.unsqueeze(-1), L).squeeze(-1)
    return (0.5 * (targets @ alpha) + torch.log(torch.diagonal(L)).sum()
            + 0.5 * n * math.log(2 * math.pi))


def curbed_gp_nlml(inputs, targets, log_lengthscales, log_signal, log_noise,
                   input_std, ls_max: float = 100.0, snr_max: float = SNR_MAX,
                   power: float = 30.0, ls_range: Optional[float] = None,
                   individual_noise=None) -> torch.Tensor:
    """PILCO's curbed NLML for one output (see :func:`curbed_nlml`)."""
    f = gp_nlml(inputs, targets, log_lengthscales, log_signal, log_noise,
                individual_noise)
    std = torch.as_tensor(input_std, dtype=log_lengthscales.dtype).reshape(-1)
    if std.numel() == 1:
        std = std.expand(log_lengthscales.numel())
    scale = math.log(ls_max) if ls_range is None else float(ls_range)
    f = f + (((log_lengthscales - torch.log(std.clamp(min=1e-6))) / scale)
             ** power).sum()
    f = f + (((log_signal - log_noise) / math.log(snr_max)) ** power).sum()
    return f


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
    """
    from ..optim.minimize import ParamVector, autograd_objective, minimize

    input_std = torch.std(op.inputs, dim=0)
    dtype = op.inputs.dtype
    for e in range(op.n_out):
        log_ell = torch.nn.Parameter(op.log_lengthscales[:, e].detach().clone())
        log_sf = torch.nn.Parameter(op.log_signal[e].detach().clone().reshape(()))
        log_sn = torch.nn.Parameter(op.log_noise[e].detach().clone().reshape(()))
        vec = ParamVector([log_ell, log_sf, log_sn])
        nigp = (op.individual_noise[:, e] if op.individual_noise.any() else None)
        evaluate = autograd_objective(
            lambda: curbed_gp_nlml(op.inputs, op.targets[:, e], log_ell, log_sf,
                                   log_sn, input_std, power=power,
                                   ls_range=ls_range, individual_noise=nigp),
            vec)
        # seed the optimiser from the current hyper-parameters, exactly like
        # PILCO (lh[:, i] is warm-started across episodes)
        with torch.no_grad():
            log_ell.clamp_(LOG_ELL_MIN, LOG_ELL_MAX)
        res = minimize(vec.flat(), evaluate, length=iterations, MFEPLS=MFEPLS)
        vec.set(res.x)
        with torch.no_grad():
            op.log_lengthscales[:, e].copy_(log_ell.detach())
            op.log_signal[e].copy_(log_sf.detach())
            op.log_noise[e].copy_(log_sn.detach())
        op.invalidate_cache()
        if verbose:
            print(f"  GP {e + 1}/{op.n_out}: {len(res.fX)} line searches, "
                  f"NLML {res.fX[0]:.6e} -> {res.final:.6e}")
    del dtype
    return op
