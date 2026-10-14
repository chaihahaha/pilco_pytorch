"""Mountain-car (continuous) scenario -- the recommended target for directed
exploration.

Why this environment
--------------------
The research summary (see ``docs/DIRECTED_EXPLORATION.md``) identifies plain
cart-pole as a *bad* test bed for directed exploration: PILCO reaches the cost
floor after a single learning episode, so the only thing an exploration bonus
can do is add cost.  Directed exploration pays off exactly when the greedy
policy's failure is *informational* rather than merely suboptimal -- the classic
case being mountain car, where the goal can only be reached by first accelerating
*away* from it.

Mountain car is also a good fit for the analytic machinery:

* the state is 2-dimensional and the dynamics are smooth apart from the velocity
  and position clamps, so a GP with ARD lengthscales models it well;
* the cost is a saturating quadratic in ``(x, v)``, so every Gaussian moment in
  the planner is closed form -- no new operator is needed;
* the difficulty is genuine: a myopic policy that always pushes towards the goal
  parks the car in the valley, so the model never sees the pumping regime and
  the greedy policy cannot discover it.

The true dynamics follow Gym's ``MountainCarContinuous-v0``::

    v' = clip(v + power * a - gravity * cos(3 x), -v_max, v_max)
    x' = clip(x + v', -1.2, 0.6)

with the goal at ``x >= 0.45``.  Instead of the sparse ``+100`` reward we use the
PILCO-style saturating cost on the distance to the goal, which is what makes the
Gaussian moment integrals closed form.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..gaussian import Gaussian
from ..ops.cost import SaturatingCostOp
from ..trajectory import StateSpec

__all__ = ["MountainCar", "make_mountaincar"]

DTYPE = torch.float64


@dataclass
class MountainCar:
    """Continuous mountain car as a PILCO scenario."""

    horizon: int = 120
    #: Scale factor on (power, gravity, v_max).  Mountain car needs
    #: ``horizon >~ 60 `` *unscaled* steps for the energy-pumping strategy to
    #: pay off inside the planning horizon; scaling the dynamics up in magnitude
    #: shortens that time without changing the difficulty (both the force and the
    #: gravity are scaled together), so the task fits in a shorter horizon.  At
    #: ``time_scale=8`` the pumping strategy reaches the goal in 61 steps and
    #: scores a ~36% lower cumulative cost than a greedy "always push right"
    #: policy over ``horizon=120`` -- the headroom directed exploration needs.
    time_scale: float = 8.0
    power: float = 0.0015
    gravity: float = 0.0025
    v_max: float = 0.07
    x_min: float = -1.2
    x_max: float = 0.6
    goal: float = 0.45
    #: cost width on the goal distance
    width: tuple = (0.3,)
    #: relative weight of the velocity term in the cost
    velocity_weight: float = 0.1
    max_u: float = 1.0
    control_noise: float = 0.0
    process_noise: float = 0.0
    n_basis: int = 12

    spec: StateSpec = field(init=False)
    cost: SaturatingCostOp = field(init=False)

    def __post_init__(self):
        self.power *= self.time_scale
        self.gravity *= self.time_scale
        self.v_max *= self.time_scale
        self.spec = StateSpec(
            state_dim=2,
            angle_indices=(),
            policy_indices=(0, 1),
            dynamics_indices=(0, 1),
            control_dim=1,
            difference_mask=(True, True),
        )
        self.cost = self.make_cost()

    # ------------------------------------------------------------------
    def make_cost(self) -> SaturatingCostOp:
        """Saturating cost on the distance to the goal.

        ``d^2 = (x - goal)^2 + w_v * v^2``; the velocity term keeps the car from
        "solving" the task by parking at the goal with residual speed and makes
        the cost landscape slightly less degenerate.
        """
        target = torch.tensor([self.goal, 0.0], dtype=DTYPE)
        W = torch.tensor([[1.0, 0.0],
                          [0.0, self.velocity_weight]], dtype=DTYPE)
        return SaturatingCostOp(target, W, self.width)

    # ------------------------------------------------------------------
    def initial_state(self, variance: float = 0.01) -> Gaussian:
        """Start at the bottom of the valley with a small uncertainty."""
        return Gaussian(torch.tensor([-0.5, 0.0], dtype=DTYPE),
                        torch.eye(2, dtype=DTYPE) * variance)

    def optimisation_state(self) -> Gaussian:
        return self.initial_state(variance=0.01)

    # ------------------------------------------------------------------
    def step_true(self, z: np.ndarray, u: float) -> np.ndarray:
        """One step of the true discrete-time dynamics."""
        x, v = float(z[0]), float(z[1])
        v = v + self.power * float(u) - self.gravity * np.cos(3.0 * x)
        v = float(np.clip(v, -self.v_max, self.v_max))
        x = float(np.clip(x + v, self.x_min, self.x_max))
        return np.array([x, v], dtype=np.float64)

    # ------------------------------------------------------------------
    def policy_numpy(self, model):
        centers = model.policy.centers.detach().numpy()
        ell = model.policy.lengthscales.detach().numpy()
        beta = model.policy.beta().detach().numpy()
        max_u = float(model.max_u[0])
        poli = np.asarray(self.spec.policy_indices)

        def control(x: np.ndarray) -> float:
            xin = np.asarray(x, dtype=np.float64).ravel()[poli]
            k = np.exp(-0.5 * (((xin[None, :] - centers) / ell[None, :]) ** 2).sum(1))
            raw = float(k @ beta[:, 0])
            return max_u * (9.0 * np.sin(raw) + np.sin(3.0 * raw)) / 8.0

        return control

    # ------------------------------------------------------------------
    def rollout_true(self, model, x0: np.ndarray, steps: int,
                     rng: np.random.Generator, with_noise: bool = True):
        control = self.policy_numpy(model)
        state = np.asarray(x0, dtype=np.float64).ravel().copy()
        states = [state.copy()]
        controls, costs = [], []
        for _ in range(steps):
            u = control(state)
            nxt = self.step_true(state, u)
            if with_noise and self.process_noise > 0:
                nxt = nxt + rng.standard_normal(2) * self.process_noise
            c = self.cost.expected(torch.as_tensor(state, dtype=DTYPE),
                                   torch.zeros(2, 2, dtype=DTYPE))
            costs.append(float(c))
            controls.append(u)
            states.append(nxt.copy())
            state = nxt
        return np.asarray(states), np.asarray(controls), np.asarray(costs)

    # ------------------------------------------------------------------
    def data_from_rollout(self, states: np.ndarray, controls: np.ndarray):
        n = controls.shape[0]
        inputs = np.column_stack([states[:n], controls.reshape(n, -1)])
        targets = states[1:n + 1] - states[:n]
        return inputs.astype(np.float64), targets.astype(np.float64)


def make_mountaincar(**kwargs) -> MountainCar:
    return MountainCar(**kwargs)
