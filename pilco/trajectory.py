"""Trajectory planning by rolling closed-form moment operators through a horizon.

The augmented-state bookkeeping is the one place where PILCO's original
structure is genuinely needed, so it is kept (and documented) rather than
abstracted away.  For a state of dimension ``D``, ``A`` angles and ``Du``
controls, one step builds a Gaussian over the slot layout::

    [ state (D) | sin/cos of angles (2A) | u_raw (Du) | u (Du) | dyn (D) ]
       idx_state      idx_trig              idx_u_raw     idx_u    idx_dyn

and fills it in four splices::

    1. TrigAugmentOp   state      -> trig       (PILCO: gTrig)
    2. policy op       poli       -> u_raw      (PILCO: congp)
    3. SaturationOp    u_raw      -> u          (PILCO: gSat)
    4. GpDynamicsOp    dyni + u   -> dyn        (PILCO: gp1d/gp0d)
    5. AffineOp  P     all        -> next state (PILCO: the final P projection)

Every arrow is a single call to :func:`~pilco.ops.base.splice`; the original
``base/propagated.py`` implements the same thing in ~140 lines of Fortran-order
index arithmetic plus a separate hand-differentiated variant.  Because the
splices are differentiable, ``plan`` is differentiable end-to-end and the
Jacobians PILCO needs for the cumulative-cost variance fall out of
``torch.func.jvp``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian import Gaussian, symmetrize
from .ops.base import AffineOp, splice
from .ops.cost import SaturatingCostOp
from .ops.fourier import SaturationOp, TrigAugmentOp
from .ops.gp import GpDynamicsOp
from .ops.policy import RbfPolicyOp

__all__ = ["StateSpec", "PilcoModel", "Trajectory", "plan", "model_rollout"]

DTYPE = torch.float64


@dataclass(frozen=True)
class StateSpec:
    """Index bookkeeping of a PILCO scenario (the role of ``plant.*`` in MATLAB).

    Attributes
    ----------
    state_dim:
        ``D``, dimension of the physical state.
    angle_indices:
        Coordinates of ``state`` that are angles and therefore augmented with
        ``sin``/``cos`` (PILCO's ``plant.angi``).
    policy_indices:
        Indices into ``[state | trig]`` fed to the controller (``plant.poli``).
    dynamics_indices:
        Indices into ``[state | trig]`` fed to the dynamics GP (``plant.dyni``).
    control_dim:
        ``Du``, dimension of the control signal.
    difference_mask:
        For each of the ``D`` GP outputs: whether the target is a *difference*
        ``x_{t+1} - x_t`` (PILCO's ``plant.difi``).
    """

    state_dim: int
    angle_indices: tuple[int, ...]
    policy_indices: tuple[int, ...]
    dynamics_indices: tuple[int, ...]
    control_dim: int
    difference_mask: tuple[bool, ...]

    # -- derived slot layout ------------------------------------------------
    @property
    def n_angles(self) -> int:
        return len(self.angle_indices)

    @property
    def trig_offset(self) -> int:
        return self.state_dim

    @property
    def u_raw_offset(self) -> int:
        return self.state_dim + 2 * self.n_angles

    @property
    def u_offset(self) -> int:
        return self.u_raw_offset + self.control_dim

    @property
    def dyn_offset(self) -> int:
        return self.u_offset + self.control_dim

    @property
    def augmented_dim(self) -> int:
        return self.dyn_offset + self.state_dim

    @property
    def aug_state_dim(self) -> int:
        """Dimension of ``[state | trig]`` (indices used by policy/dynamics)."""
        return self.state_dim + 2 * self.n_angles

    def _range(self, lo: int, hi: int, dtype=torch.long) -> torch.Tensor:
        return torch.arange(lo, hi, dtype=dtype)

    @property
    def idx_state(self) -> torch.Tensor:
        return self._range(0, self.state_dim)

    @property
    def idx_trig(self) -> torch.Tensor:
        return self._range(self.trig_offset, self.u_raw_offset)

    @property
    def idx_u_raw(self) -> torch.Tensor:
        return self._range(self.u_raw_offset, self.u_offset)

    @property
    def idx_u(self) -> torch.Tensor:
        return self._range(self.u_offset, self.dyn_offset)

    @property
    def idx_dyn(self) -> torch.Tensor:
        return self._range(self.dyn_offset, self.augmented_dim)

    @property
    def idx_gp_input(self) -> torch.Tensor:
        """GP query points ``[dyni | u]`` inside the augmented vector."""
        return torch.cat([torch.as_tensor(self.dynamics_indices),
                          self.idx_u])

    def projection(self) -> torch.Tensor:
        """PILCO's ``P``: ``next_state = dyn + state`` on the difference coords."""
        P = torch.zeros(self.state_dim, self.augmented_dim, dtype=DTYPE)
        P[:, self.dyn_offset:] = torch.eye(self.state_dim, dtype=DTYPE)
        for k, is_diff in enumerate(self.difference_mask):
            if is_diff:
                P[k, k] = 1.0
        return P


class PilcoModel(nn.Module):
    """Everything PILCO needs to plan: dynamics GP, controller, cost, indexing."""

    def __init__(self, spec: StateSpec, dynamics: GpDynamicsOp,
                 policy: RbfPolicyOp, cost: SaturatingCostOp,
                 max_u: Optional[Sequence[float]] = None,
                 control_noise: float = 0.0):
        super().__init__()
        self.spec = spec
        self.dynamics = dynamics
        self.policy = policy
        self.cost = cost
        mu = max_u if max_u is not None else [1.0] * spec.control_dim
        self.register_buffer("max_u", torch.as_tensor(list(mu), dtype=DTYPE))
        self.control_noise = float(control_noise)
        self.trig_op = TrigAugmentOp(spec.angle_indices, n_in=spec.state_dim)
        self.sat_op = SaturationOp(spec.control_dim,
                                   list(range(spec.control_dim)), self.max_u)
        self.proj_op = AffineOp(spec.projection())

    # ------------------------------------------------------------------
    # NOTE: ``torch.compile`` on the one-step map was tried and removed.
    # Measured on this machine: it produced identical results (cost difference
    # 0.0) but gave no end-to-end speedup at the ``plan`` level (385 ms compiled
    # vs 381 ms eager for H=12, excluding the one-off compile cost), and it
    # crashed in Inductor's C++ backend during the *backward* pass
    # (``assert mode is None`` in torch/_inductor/codegen/cpp.py) -- which a
    # try/except around the forward cannot catch, because the backward is
    # triggered later, inside ``torch.autograd.grad``.  See docs/ARCHITECTURE.md.

    def step_augmented(self, mean: torch.Tensor, cov: torch.Tensor,
                       track_jacobian: bool = False):
        """One PILCO step.

        Returns ``(next_mean, next_cov, augmented_mean, augmented_cov)`` where
        the augmented pair is the full slot Gaussian *before* the projection
        (that is where the GP query points live).

        If ``track_jacobian`` is true, a fifth value is returned:
        ``J = d next_mean / d mean``.  It is assembled from the ``cross`` blocks
        the operators already produce -- the conditional-mean map of an operator
        is ``S_in^{-1} Cov[x_in, out] = C``, so ``d out_mean / d in_mean =
        C^T`` -- which means the Jacobian costs nothing beyond the forward pass.
        That is the same linearisation PILCO uses everywhere else, and it is
        checked against ``torch.func.jvp`` in ``tests/test_jacobian.py``.
        """
        spec = self.spec
        n = spec.augmented_dim
        pad = n - spec.state_dim
        aug_m = F.pad(mean, (0, pad))
        aug_S = F.pad(symmetrize(cov), (0, pad, 0, pad))
        Dm = None
        if track_jacobian:
            Dm = torch.zeros(n, spec.state_dim, dtype=mean.dtype, device=mean.device)
            Dm = Dm.index_copy(0, spec.idx_state,
                               torch.eye(spec.state_dim, dtype=mean.dtype,
                                         device=mean.device))

        def _splice(mean, cov, op, in_idx, out_idx):
            nonlocal Dm
            mean, cov, res = splice(mean, cov, op, in_idx, out_idx)
            if Dm is not None:
                ii = torch.as_tensor(list(in_idx)) if not isinstance(in_idx, torch.Tensor) else in_idx
                oi = torch.as_tensor(list(out_idx)) if not isinstance(out_idx, torch.Tensor) else out_idx
                Dm = Dm.index_copy(0, oi, res.cross.transpose(0, 1) @ Dm[ii])
            return mean, cov

        # 1) sin/cos augmentation
        aug_m, aug_S = _splice(aug_m, aug_S, self.trig_op, spec.idx_state,
                               spec.idx_trig)
        # 2) controller on [state | trig]
        aug_m, aug_S = _splice(aug_m, aug_S, self.policy, spec.policy_indices,
                               spec.idx_u_raw)
        if self.control_noise > 0.0:
            idx = spec.idx_u_raw
            k = idx.numel()
            noise = (self.control_noise ** 2) * torch.eye(
                k, dtype=mean.dtype, device=mean.device)
            aug_S = symmetrize(aug_S.index_put(
                (idx.repeat_interleave(k), idx.repeat(k)),
                (aug_S[idx][:, idx] + noise).reshape(-1)))
        # 3) saturation
        aug_m, aug_S = _splice(aug_m, aug_S, self.sat_op, spec.idx_u_raw,
                               spec.idx_u)
        # 4) dynamics GP on [dyni | u]
        aug_m, aug_S = _splice(aug_m, aug_S, self.dynamics, spec.idx_gp_input,
                               spec.idx_dyn)
        # 5) project to the next state
        P = self.proj_op.weight
        next_m = P @ aug_m
        next_S = symmetrize(P @ aug_S @ P.transpose(0, 1))
        if Dm is not None:
            return next_m, next_S, aug_m, aug_S, P @ Dm
        return next_m, next_S, aug_m, aug_S

    # ------------------------------------------------------------------
    def control(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Mean (deterministic) control at a state -- used for evaluation."""
        _, _, aug_m, _ = self.step_augmented(mean, cov)
        return aug_m[self.spec.idx_u]

    def state_dict_policy(self) -> dict:
        layer = self.policy
        return {"centers": layer.centers.detach().clone(),
                "targets": layer.targets.detach().clone(),
                "lengthscales": layer.lengthscales.detach().clone()}


@dataclass
class Trajectory:
    """Result of :func:`plan`.

    Attributes
    ----------
    states:
        ``H + 1`` :class:`Gaussian` objects; ``states[0]`` is the initial
        distribution and ``states[t]`` the predicted state at time ``t``.
    augmented_mean / augmented_cov:
        ``(H, n_aug)`` and ``(H, n_aug, n_aug)``: the full augmented Gaussian
        before each transition (where the GP query points live).
    costs / cost_vars:
        ``(H,)`` expected value and variance of the immediate cost of each
        transition (PILCO evaluates the cost *after* the transition).
    jacobians:
        ``H`` matrices ``J_t = d mean_{t+1} / d mean_t`` from forward-mode
        differentiation; needed for the cross-timestep covariances.
    """

    states: list[Gaussian]
    augmented_mean: torch.Tensor
    augmented_cov: torch.Tensor
    costs: torch.Tensor
    cost_vars: torch.Tensor
    jacobians: list[torch.Tensor]
    gamma: float = 1.0

    @property
    def horizon(self) -> int:
        return len(self.states) - 1

    def discounted_costs(self) -> torch.Tensor:
        disc = torch.as_tensor(
            [self.gamma ** (t + 1) for t in range(self.horizon)],
            dtype=self.costs.dtype, device=self.costs.device)
        return self.costs * disc

    def value(self) -> torch.Tensor:
        """Expected discounted cumulative cost (PILCO's objective)."""
        return self.discounted_costs().sum()

    def gp_inputs(self, spec: StateSpec) -> tuple[torch.Tensor, torch.Tensor]:
        """Means/covariances of the GP query points ``[dyni | u]`` per step."""
        idx = spec.idx_gp_input
        means = self.augmented_mean.index_select(1, idx)
        covs = self.augmented_cov.index_select(1, idx).index_select(2, idx)
        return means, covs


def plan(model: PilcoModel, initial: Gaussian, horizon: int,
         gamma: float = 1.0, with_jacobians: bool = True) -> Trajectory:
    """Roll the moment operators forward for ``horizon`` steps."""
    spec = model.spec
    m, S = initial.mean, initial.cov
    states = [Gaussian(m, S)]
    aug_means, aug_covs = [], []
    costs, cost_vars = [], []
    jacobians = []

    for _ in range(horizon):
        if with_jacobians:
            jacobians.append(mean_jacobian(m, S, model))
        next_m, next_S, aug_m, aug_S = model.step_augmented(m, S)
        aug_means.append(aug_m)
        aug_covs.append(aug_S)
        c, v = model.cost.moments(next_m, next_S)
        costs.append(c)
        cost_vars.append(v)
        states.append(Gaussian(next_m, next_S))
        m, S = next_m, next_S

    empty_m = torch.zeros(0, spec.augmented_dim, dtype=DTYPE)
    empty_S = torch.zeros(0, spec.augmented_dim, spec.augmented_dim, dtype=DTYPE)
    empty_c = torch.zeros(0, dtype=DTYPE)
    return Trajectory(
        states=states,
        augmented_mean=torch.stack(aug_means) if aug_means else empty_m,
        augmented_cov=torch.stack(aug_covs) if aug_covs else empty_S,
        costs=torch.stack(costs) if costs else empty_c,
        cost_vars=torch.stack(cost_vars) if cost_vars else empty_c,
        jacobians=jacobians,
        gamma=gamma,
    )


def mean_jacobian(m: torch.Tensor, S: torch.Tensor,
                  model: PilcoModel) -> torch.Tensor:
    """``J = d mean_{t+1} / d mean_t`` at fixed input covariance.

    Implemented with ``torch.func.jacfwd`` -- forward-mode over the *actual*
    forward computation.  Two remarks on why this is the right tool:

    * The Jacobian is **not** simply the chain of the operators' ``cross``
      blocks.  ``cross`` gives ``d mean_out / d mean_in``, but each operator's
      mean also depends on the input covariance (e.g.
      ``E[sin x] = e^{-v/2} sin m``), and the intermediate covariances depend on
      the state mean.  PILCO carries this as the separate ``dMds``/``dSdm``
      blocks; getting it from autograd removes an entire class of hand-derived
      Jacobians.
    * ``jacfwd`` is vmap-over-jvp, so the primal computation is shared across the
      ``D`` tangent directions and it is differentiable with respect to the
      policy parameters (needed because the cumulative-cost variance objective
      contains ``J``).
    """
    from torch.func import jacfwd

    def f(mm):
        return model.step_augmented(mm, S)[0]

    return jacfwd(f)(m)


def mean_jacobian_jvp(m: torch.Tensor, S: torch.Tensor,
                      model: PilcoModel) -> torch.Tensor:
    """Reference implementation of :func:`mean_jacobian` via ``torch.func.jvp``.

    Forward-mode differentiation of the *actual* forward computation.  Kept
    because it validates the cheap ``cross``-block assembly without relying on
    any of the same algebra.
    """
    from torch.func import jvp

    def f(mm):
        return model.step_augmented(mm, S)[0]

    D = m.numel()
    eye = torch.eye(D, dtype=m.dtype, device=m.device)
    cols = [jvp(f, (m,), (eye[i],))[1] for i in range(D)]
    return torch.stack(cols, dim=1)


def model_rollout(model: PilcoModel, x0: torch.Tensor, steps: int,
                  noise: float = 0.0, generator: Optional[torch.Generator] = None):
    """Roll the *learned* model forward at a deterministic state (no planning).

    Returns the tensor of applied controls and the state trajectory.
    """
    x = x0.reshape(-1).clone()
    xs = [x.clone()]
    us = []
    D = x.numel()
    for _ in range(steps):
        zero = torch.zeros(D, D, dtype=x.dtype, device=x.device)
        _, _, aug_m, _ = model.step_augmented(x, zero)
        u = aug_m[model.spec.idx_u]
        us.append(u)
        trig = model.trig_op.moments(x, zero).mean
        aug = torch.cat([x, trig])
        inp = torch.cat([aug[list(model.spec.dynamics_indices)], u])
        pred = model.dynamics.moments(
            inp, torch.zeros(inp.numel(), inp.numel(), dtype=x.dtype,
                             device=x.device)).mean
        if noise > 0.0:
            pred = pred + noise * torch.randn(pred.shape, dtype=x.dtype,
                                              device=x.device, generator=generator)
        P = model.proj_op.weight
        x = P @ torch.cat([aug, u, u, pred])
        xs.append(x.clone())
    return torch.stack(us) if us else torch.zeros(0, model.spec.control_dim,
                                                  dtype=x.dtype), torch.stack(xs)
