"""Cart-pole swing-up scenario.

Everything scenario-specific lives here; the algorithm only sees a
:class:`~pilco.trajectory.PilcoModel`.  The physical setting matches
``pilco_python/scenarios/cartPole``:

* state ``[x, dx, dtheta, theta]``, control ``u`` in ``[-10, 10]`` N, ``dt=0.1``
* angles are augmented with ``sin``/``cos`` before the cost and policy
* the GP predicts **state differences** ``x_{t+1} - x_t``
* cost ``1 - exp(-1/2 d^2 / width^2)`` with the tip-position distance
  ``d^2 = (x + l sin theta)^2 + l^2 (cos theta + 1)^2``, ``width = 0.25``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ..gaussian import Gaussian
from ..models.gp import build_dynamics_op
from ..ops.cost import TrigAugmentedCostOp
from ..ops.gp import GpDynamicsOp
from ..ops.policy import RbfPolicyOp
from ..trajectory import PilcoModel, StateSpec

__all__ = ["CartPole", "make_cartpole"]

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# true (nonlinear) dynamics -- identical to pilco_python/scenarios/cartPole
# ---------------------------------------------------------------------------
def cartpole_dynamics(t: float, z: np.ndarray, f: float) -> np.ndarray:
    """Continuous-time cart-pole dynamics (Deisenroth thesis, Appendix C)."""
    length = 0.5      # [m]
    mp = 0.5          # [kg] pendulum mass
    mc = 0.5          # [kg] cart mass
    b = 0.1           # [N/m/s] friction
    g = 9.82          # [m/s^2]
    z = np.asarray(z, dtype=np.float64).ravel()
    s, c = np.sin(z[3]), np.cos(z[3])
    den1 = 4.0 * (mc + mp) - 3.0 * mp * c * c
    den2 = 4.0 * length * (mp + mc) - 3.0 * mp * length * c * c
    dz = np.zeros(4)
    dz[0] = z[1]
    dz[1] = (2 * mp * length * z[2] ** 2 * s + 3 * mp * g * s * c
             + 4 * f - 4 * b * z[1]) / den1
    dz[2] = (-3 * mp * length * z[2] ** 2 * s * c - 6 * (mc + mp) * g * s
             - 6 * (f - b * z[1]) * c) / den2
    dz[3] = z[2]
    return dz


@dataclass
class CartPole:
    """Cart-pole scenario: spec, cost, true dynamics and data plumbing."""

    horizon: int = 40
    dt: float = 0.1
    length: float = 0.5
    width: tuple = (0.25,)
    max_u: float = 10.0
    control_noise: float = 0.0
    process_noise: float = 0.01
    n_basis: int = 12

    spec: StateSpec = field(init=False)
    cost: TrigAugmentedCostOp = field(init=False)

    def __post_init__(self):
        self.spec = StateSpec(
            state_dim=4,
            angle_indices=(3,),
            policy_indices=(0, 1, 2, 4, 5),
            dynamics_indices=(0, 1, 2, 4, 5),
            control_dim=1,
            difference_mask=(True, True, True, True),
        )
        self.cost = self.make_cost()

    # ------------------------------------------------------------------
    def make_cost(self) -> TrigAugmentedCostOp:
        """The cart-pole cost on the trig-augmented state (PILCO's ``loss_cp``)."""
        ell = self.length
        target = torch.tensor([0.0, 0.0, 0.0, np.pi, 0.0, -1.0], dtype=DTYPE)
        Q = torch.zeros(6, 6, dtype=DTYPE)
        Q[0, 0] = 1.0
        Q[0, 4] = ell
        Q[4, 0] = ell
        Q[4, 4] = ell ** 2
        Q[5, 5] = ell ** 2
        return TrigAugmentedCostOp((3,), 4, target, Q, self.width)

    # ------------------------------------------------------------------
    def initial_state(self, variance: float = 0.01) -> Gaussian:
        """``mu0 = 0``, ``S0 = diag(0.01)`` (state std 0.1)."""
        return Gaussian(torch.zeros(4, dtype=DTYPE),
                        torch.eye(4, dtype=DTYPE) * variance)

    def optimisation_state(self) -> Gaussian:
        """The distribution PILCO plans from (``mu0Sim``/``S0Sim``)."""
        return self.initial_state(variance=0.01)

    # ------------------------------------------------------------------
    def step_true(self, z: np.ndarray, u: float) -> np.ndarray:
        """One ``dt`` step of the true system with a zero-order-hold control."""
        sol = solve_ivp(lambda t, y: cartpole_dynamics(t, y, u), (0.0, self.dt),
                        np.asarray(z, dtype=np.float64).ravel(), method="RK45",
                        t_eval=[self.dt], rtol=1e-12, atol=1e-12)
        return sol.y[:, -1]

    # ------------------------------------------------------------------
    def policy_numpy(self, model: PilcoModel):
        """Return a fast NumPy evaluator of the controller."""
        centers = model.policy.centers.detach().numpy()
        ell = model.policy.lengthscales.detach().numpy()
        beta = model.policy.beta().detach().numpy()
        max_u = float(model.max_u[0])
        poli = np.asarray(self.spec.policy_indices)

        def control(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=np.float64).ravel()
            aug = np.concatenate([x, [np.sin(x[3]), np.cos(x[3])]])
            xin = aug[poli]
            k = np.exp(-0.5 * (((xin[None, :] - centers) / ell[None, :]) ** 2).sum(1))
            raw = float(k @ beta[:, 0])
            return max_u * (9.0 * np.sin(raw) + np.sin(3.0 * raw)) / 8.0

        return control

    # ------------------------------------------------------------------
    def rollout_true(self, model: PilcoModel, x0: np.ndarray, steps: int,
                     rng: np.random.Generator, with_noise: bool = True):
        """Roll out the *true* environment, returning states/controls/costs.

        Matches ``base/rollout.py``: process noise is added to every component
        of the successor state, and the noisy states are what get logged.
        """
        control = self.policy_numpy(model)
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        state = x0 + rng.standard_normal(4) * self.process_noise if with_noise else x0.copy()
        states = [state.copy()]
        controls, costs = [], []
        for _ in range(steps):
            u = control(state)
            nxt = self.step_true(state, u)
            if with_noise:
                nxt = nxt + rng.standard_normal(4) * self.process_noise
            c = self.cost.expected(
                torch.as_tensor(state, dtype=DTYPE),
                torch.zeros(4, 4, dtype=DTYPE))
            costs.append(float(c))
            controls.append(u)
            states.append(nxt.copy())
            state = nxt
        return np.asarray(states), np.asarray(controls), np.asarray(costs)

    # ------------------------------------------------------------------
    def data_from_rollout(self, states: np.ndarray, controls: np.ndarray):
        """Turn a trajectory into GP inputs/targets (``trainDynModel.py``).

        Inputs ``z_t = [x, dx, dtheta, sin(theta), cos(theta), u_t]`` (6 dims),
        targets are the **state differences** ``x_{t+1} - x_t`` (4 dims).
        """
        n = controls.shape[0]
        s = states[:n]
        trig = np.column_stack([np.sin(s[:, 3]), np.cos(s[:, 3])])
        aug = np.column_stack([s, trig])                       # (n, 6)
        dyni = np.asarray(self.spec.dynamics_indices)
        inputs = np.column_stack([aug[:, dyni], controls.reshape(n, -1)])
        targets = states[1:n + 1] - states[:n]
        return inputs.astype(np.float64), targets.astype(np.float64)


def make_cartpole(**kwargs) -> CartPole:
    return CartPole(**kwargs)
