"""Train PILCO on cart-pole, then render the learned swing-up as an animation.

The animation is not decoration: it is the evidence for the claim that the
controller swaps a hanging pendulum for an inverted one.  It therefore shows the
three quantities that decide the question, side by side:

* the cart-pendulum in Cartesian space (cart on a rail, pendulum from the pivot),
  with the centre of mass of the pole drawn as a disc -- ``theta = pi`` is up;
* ``theta(t)`` in degrees against the upright line -- swing-up is a *rise and
  hold*, so a spike that falls back is immediately visible;
* the per-step task cost ``1 - exp(-d^2 / 2 w^2)`` and its running total --
  PILCO's summed cost over the horizon is exactly the total plotted here.

The dynamics used for the animation are the true ones (``solve_ivp``, RK45,
``rtol=atol=1e-12``) with the zero-order-hold control of the learning loop, but
integrated at ``--substeps`` finer resolution so that the motion is smooth.
Process noise is off, so what is shown is the *mean* behaviour the controller was
optimised for; the noisy cost it actually pays is reported alongside.

Usage::

    # train (cached) and render both formats
    python scripts/make_animation.py --episodes 8 --seed 0

    # render from an already trained policy
    python scripts/make_animation.py --no-train

Outputs land in ``results/``: ``cartpole_swingup.gif``, ``.mp4`` and the static
diagnostic ``cartpole_swingup_timeseries.png``.  ``results/`` is gitignored.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Circle, Rectangle                 # noqa: E402
from scipy.integrate import solve_ivp                        # noqa: E402

from pilco.pilco import PilcoConfig, learn                   # noqa: E402
from pilco.scenarios.cartpole import cartpole_dynamics, make_cartpole  # noqa: E402


# ---------------------------------------------------------------------------
# training / cache
# ---------------------------------------------------------------------------
def train(episodes: int, seed: int, horizon: int, n_basis: int,
          gp_iterations: int, policy_iterations: int, cache: str):
    """Run the faithful protocol and dump the learned controller."""
    scenario = make_cartpole(horizon=horizon, n_basis=n_basis)
    config = PilcoConfig(
        horizon=horizon, episodes=episodes, seed=seed,
        gp_iterations=gp_iterations, policy_iterations=policy_iterations,
        # the two settings that make the loop reproduce the reference
        gp_optimizer="pilco", policy_optimizer="pilco", legacy_rng=True,
        verbose=True,
    )
    t0 = time.time()
    model, logs = learn(scenario, config)
    elapsed = time.time() - t0
    costs = [float(np.sum(l.realised_costs)) for l in logs]
    print(f"\ntrained {episodes} episodes in {elapsed / 60:.1f} min; "
          f"costs={[round(c, 2) for c in costs]}")

    pol = model.policy
    np.savez(
        cache,
        centers=pol.centers.detach().numpy(),
        targets=pol.targets.detach().numpy(),
        beta=pol.beta().detach().numpy(),
        lengthscales=torch.exp(pol.log_lengthscales).detach().numpy(),
        noise=pol.noise,
        signal=pol.signal,
        max_u=float(model.max_u[0]),
        policy_indices=np.asarray(model.spec.policy_indices),
        costs=np.asarray(costs),
        elapsed=elapsed,
    )
    print(f"controller saved to {cache}")
    return costs


# ---------------------------------------------------------------------------
# fast NumPy controller, identical to CartPole.policy_numpy
# ---------------------------------------------------------------------------
class Controller:
    """NumPy mirror of :meth:`CartPole.policy_numpy`.

    The RBF network's *weights* are ``beta = (K + sigma_n^2 I)^{-1} T``, not the
    pseudo-targets ``T`` -- using ``T`` directly (an easy mistake) drops the
    regularisation and produces a completely different, non-working controller.
    ``beta`` is recomputed here from the cached centres/lengthscales so the cache
    only needs to store the parameters, and cross-checked against the value the
    training run saved.
    """

    def __init__(self, path: str):
        d = np.load(path)
        self.centers = d["centers"]
        self.targets = d["targets"]
        self.ell = d["lengthscales"]
        self.max_u = float(d["max_u"])
        self.poli = np.asarray(d["policy_indices"])
        self.costs = np.atleast_1d(d["costs"])
        self.noise = float(d["noise"]) if "noise" in d else 0.01
        self.signal = float(d["signal"]) if "signal" in d else 1.0

        # beta = (K + sigma_n^2 I)^{-1} T, exactly as RbfPolicyOp.beta()
        c = self.centers
        sq = ((c[:, None, :] - c[None, :, :]) / self.ell[None, None, :]) ** 2
        K = self.signal * np.exp(-0.5 * sq.sum(-1))
        A = K + (self.noise ** 2) * np.eye(c.shape[0])
        self.beta = np.linalg.solve(A, self.targets)
        if "beta" in d:
            # sanity check only: numpy and torch solve the same system with
            # different pivoting, so agreement is ~1e-12 relative, not exact
            err = np.max(np.abs(self.beta - np.asarray(d["beta"])))
            if err > 1e-8:
                raise RuntimeError(f"beta mismatch against the cached value: {err:.2e}")

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        aug = np.concatenate([x, [np.sin(x[3]), np.cos(x[3])]])
        xin = aug[self.poli]
        k = np.exp(-0.5 * (((xin[None, :] - self.centers) / self.ell[None, :]) ** 2).sum(1))
        raw = float(k @ self.beta[:, 0])
        return self.max_u * (9.0 * np.sin(raw) + np.sin(3.0 * raw)) / 8.0


def step_cost(state: np.ndarray, length: float, width: float) -> float:
    """PILCO's cart-pole cost ``1 - exp(-d^2 / 2 w^2)`` at one state."""
    x, th = state[0], state[3]
    d2 = (x + length * np.sin(th)) ** 2 + length ** 2 * (np.cos(th) + 1.0) ** 2
    return float(1.0 - np.exp(-0.5 * d2 / width ** 2))


def simulate(control, x0, steps, dt, length, width, substeps, u_max=10.0):
    """ZOH rollout of the true dynamics, recorded at ``substeps`` finer grid."""
    state = np.asarray(x0, dtype=np.float64).ravel().copy()
    t = [0.0]
    states = [state.copy()]
    controls = []
    step_costs = [step_cost(state, length, width)]
    us = []
    for _ in range(steps):
        u = float(np.clip(control(state), -u_max, u_max))
        us.append(u)
        te = np.linspace(0.0, dt, substeps + 1)[1:]
        sol = solve_ivp(lambda tt, y: cartpole_dynamics(tt, y, u), (0.0, dt),
                        state, method="RK45", t_eval=te,
                        rtol=1e-11, atol=1e-12)
        for k in range(substeps):
            state = sol.y[:, k].copy()
            t.append(t[-1] + dt / substeps)
            states.append(state.copy())
            step_costs.append(step_cost(state, length, width))
        controls.append(u)
    return (np.asarray(t), np.asarray(states), np.asarray(us),
            np.asarray(step_costs))


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def draw(ax, state, length, xlim, trail, tip_trail, upright_tol):
    ax.clear()
    ax.set_xlim(*xlim)
    # the pendulum tip sits at ``pivot + l (sin th, -cos th)``: hanging down at
    # ``theta = 0``, upright at ``theta = pi`` -- which is exactly the geometry
    # behind the cost's ``l^2 (cos theta + 1)^2`` term.
    ax.set_ylim(-length - 0.25, length + 0.42)
    ax.set_aspect("equal")
    ax.axhline(0.0, color="#9aa0a6", lw=3, zorder=1)
    ax.set_yticks([])
    ax.set_xlabel("x [m]")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    x, th = float(state[0]), float(state[3])
    half_w, height = 0.22, 0.13
    # goal region: any pole tip inside this disc counts as "up"
    ax.add_patch(Circle((0.0, length), upright_tol, facecolor="#2e7d32",
                        alpha=0.07, edgecolor="#2e7d32", ls=":", lw=1.1, zorder=0))
    ax.text(0.0, length + upright_tol + 0.05, "goal", ha="center", va="bottom",
            fontsize=8, color="#2e7d32")

    tip = np.array([x + length * np.sin(th), -length * np.cos(th)])
    if len(tip_trail) > 2:
        tt = np.asarray(tip_trail)
        ax.plot(tt[:, 0], tt[:, 1], color="#1a73e8", alpha=0.28, lw=1.0, zorder=2)
    if len(trail) > 2:
        tr = np.asarray(trail)
        ax.plot(tr, np.zeros_like(tr), color="#1a73e8", alpha=0.18, lw=6,
                solid_capstyle="round", zorder=2)

    ax.add_patch(Rectangle((x - half_w, 0.0), 2 * half_w, height,
                           facecolor="#5f6368", edgecolor="#202124",
                           lw=1.2, zorder=4))
    ax.add_patch(Circle((x, 0.0), 0.045, facecolor="#202124", zorder=6))
    ax.plot([x, tip[0]], [0.0, tip[1]], color="#202124", lw=5,
            solid_capstyle="round", zorder=5)
    mass = 0.5 * (np.array([x, 0.0]) + tip)
    ax.add_patch(Circle(mass, 0.06, facecolor="#d93025", edgecolor="#202124",
                        lw=1.0, zorder=7))
    ax.add_patch(Circle(tip, 0.075, facecolor="#fbbc04", edgecolor="#202124",
                        lw=1.0, zorder=7))
    # angle arc from the downward direction, so "swing-up" reads as a sweep
    arc = np.linspace(0.0, th, 40)
    ax.plot(x + 0.16 * np.sin(arc), -0.16 * np.cos(arc), color="#1a73e8", lw=1.4,
            zorder=3)
    ax.annotate("", xy=(x + 0.16 * np.sin(th), -0.16 * np.cos(th)),
                xytext=(x, 0.0),
                arrowprops=dict(arrowstyle="->", color="#1a73e8", lw=1.0))


def render(t, states, us, step_costs, costs, length, width, horizon, dt,
           substeps, gif_path, mp4_path, fps, verdict):
    """Render one frame per sample and assemble GIF + MP4.

    Layout: the spatial view spans the full width on top (a wide strip would make
    an equal-aspect drawing tiny), the two time series share the row below.
    """
    import imageio.v2 as imageio

    theta = states[:, 3]
    x = states[:, 0]
    # Keep the camera on the action: a successful swing-up stays near the origin,
    # so a fixed-width window keeps the pole large.  Widen only if it must.
    xc = 0.5 * (float(x.min()) + float(x.max()))
    half = max(1.25, float(np.abs(x - xc).max()) + 0.35)
    xlim = (xc - half, xc + half)
    upright_tol = float(width)
    per_step = step_costs[::substeps][:horizon]
    cum = np.cumsum(per_step)
    t_step = np.arange(per_step.size) * dt
    # the angle panel must show the full swing (cart-pole starts 180 deg off and
    # overshoots), so size it from the data rather than a fixed range
    ang_lim = max(190.0, float(np.abs(np.degrees(theta) - 180.0).max()) * 1.08)

    fig = plt.figure(figsize=(10.4, 7.2), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0],
                          left=0.085, right=0.975, top=0.895, bottom=0.095,
                          wspace=0.20, hspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    ax_ang = fig.add_subplot(gs[1, 0])
    ax_cost = fig.add_subplot(gs[1, 1])

    n_sub = len(t)
    hold = int(0.8 * fps)                       # freeze the final pose
    frames = []

    for k in range(n_sub + hold):
        kk = min(k, n_sub - 1)
        step = min(kk // substeps, per_step.size - 1)
        fig.suptitle(
            f"cart-pole swing-up, learned controller   |   {verdict}   |   "
            f"t = {t[kk]:.2f} s   |   $\\theta$ = {np.degrees(theta[kk]) - 180:+.1f}$^\\circ$ "
            f"from upright   |   cost {step_costs[kk]:.3f}",
            fontsize=11, y=0.972)
        draw(ax, states[kk], length, xlim, x[:kk + 1],
             np.column_stack([x[:kk + 1] + length * np.sin(theta[:kk + 1]),
                              -length * np.cos(theta[:kk + 1])]), upright_tol)

        ax_ang.clear()
        ax_ang.axhline(0.0, color="#2e7d32", lw=1.2, ls="--")
        ax_ang.axhline(20.0, color="#2e7d32", lw=0.8, ls=":")
        ax_ang.axhline(-20.0, color="#2e7d32", lw=0.8, ls=":")
        ax_ang.plot(t, np.degrees(theta) - 180.0, color="#5f6368", lw=1.2)
        ax_ang.plot(t[:kk + 1], np.degrees(theta[:kk + 1]) - 180.0,
                    color="#1a73e8", lw=2.0)
        ax_ang.plot([t[kk]], [np.degrees(theta[kk]) - 180.0], "o",
                    color="#d93025", ms=5)
        ax_ang.set_ylabel(r"$\theta - \pi$  [$^\circ$]")
        ax_ang.set_ylim(-ang_lim, ang_lim)
        ax_ang.set_xlim(0.0, t[-1])
        ax_ang.grid(alpha=0.25)
        for s_ in ("top", "right"):
            ax_ang.spines[s_].set_visible(False)
        ax_ang.set_title("angle (0$^\\circ$ = upright, band $\\pm$20$^\\circ$)",
                         fontsize=9)

        ax_cost.clear()
        ax_cost.plot(t_step, cum, color="#5f6368", lw=1.2, label="running total")
        ax_cost.plot(t_step[:step + 1], cum[:step + 1], color="#1a73e8", lw=2.0)
        ax_cost.set_xlim(0.0, horizon * dt)
        ax_cost.set_ylim(0.0, max(1.5, float(cum.max()) * 1.08))
        ax_cost.set_xlabel("t [s]")
        ax_cost.set_ylabel("cumulative cost")
        ax_cost.grid(alpha=0.25)
        ax_cost.text(0.97, 0.06,
                     f"total = {cum.max():.2f}   "
                     f"(training run: {', '.join(f'{c:.1f}' for c in costs)})",
                     transform=ax_cost.transAxes, ha="right", va="bottom",
                     fontsize=7.0, color="#5f6368")
        for s_ in ("top", "right"):
            ax_cost.spines[s_].set_visible(False)
        ax_cost.set_title("cost $1-\\exp(-d^2/2w^2)$, summed", fontsize=9)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[:, :, :3].copy())

    plt.close(fig)

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=1.0 / fps, loop=0)
    print(f"wrote {gif_path}  ({len(frames)} frames, {len(frames) / fps:.1f} s @ {fps} fps)")

    if mp4_path:
        tmp = os.path.join(os.path.dirname(mp4_path) or ".", "_anim_frames")
        os.makedirs(tmp, exist_ok=True)
        for i, f in enumerate(frames):
            imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), f)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", os.path.join(tmp, "f%05d.png"),
               "-vcodec", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "20", mp4_path]
        subprocess.run(cmd, check=True)
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
        print(f"wrote {mp4_path}")
    return frames


def timeseries_figure(t, states, us, step_costs, costs, length, width, horizon,
                      dt, substeps, path, seed, title_note):
    """Static summary: the quantitative version of the animation."""
    theta = states[:, 3]
    x = states[:, 0]
    d = np.sqrt((x + length * np.sin(theta)) ** 2
                + length ** 2 * (np.cos(theta) + 1.0) ** 2)
    per_step = step_costs[::substeps][:horizon]
    t_step = np.arange(per_step.size) * dt
    t_u = np.arange(len(us)) * dt

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.4), sharex=True,
                             constrained_layout=True)
    axes[0].plot(t, np.degrees(theta) - 180.0, color="#1a73e8", lw=1.8)
    axes[0].axhline(0.0, color="#2e7d32", ls="--", lw=1.0)
    axes[0].axhline(20.0, color="#2e7d32", ls=":", lw=0.9)
    axes[0].axhline(-20.0, color="#2e7d32", ls=":", lw=0.9)
    axes[0].set_ylabel(r"$\theta-\pi$ [$^\circ$]")
    axes[0].set_title(f"cart-pole swing-up, learned controller (seed {seed})"
                      + (f" — {title_note}" if title_note else ""), fontsize=10)
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, d, color="#d93025", lw=1.8, label="tip distance $d$")
    axes[1].axhline(width, color="#5f6368", ls=":", lw=1.0,
                    label=f"cost width $w={width}$")
    axes[1].set_ylabel("$d$ [m]")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.25)

    axes[2].step(t_u, us, where="post", color="#f9ab00", lw=1.4, label="control $u$ [N]")
    axes[2].set_ylabel("$u$ [N]")
    axes[2].set_xlabel("t [s]")
    axes[2].grid(alpha=0.25)
    ax2 = axes[2].twinx()
    ax2.plot(t_step, np.cumsum(per_step), color="#5f6368", lw=1.4)
    ax2.set_ylabel("cumulative cost", color="#5f6368")
    ax2.tick_params(axis="y", colors="#5f6368")
    axes[2].text(0.98, 0.03, "yellow: control $u(t)$    grey: cumulative cost",
                 transform=axes[2].transAxes, fontsize=7.0, color="#5f6368",
                 ha="right", va="bottom")

    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--n-basis", type=int, default=12)
    ap.add_argument("--gp-iterations", type=int, default=150)
    ap.add_argument("--policy-iterations", type=int, default=150)
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-train", action="store_true",
                    help="reuse results/cartpole_swingup_policy.npz")
    ap.add_argument("--cache", default=os.path.join(_ROOT, "results",
                                                    "cartpole_swingup_policy.npz"))
    ap.add_argument("--gif", default=os.path.join(_ROOT, "results",
                                                  "cartpole_swingup.gif"))
    ap.add_argument("--mp4", default=os.path.join(_ROOT, "results",
                                                  "cartpole_swingup.mp4"))
    ap.add_argument("--png", default=os.path.join(_ROOT, "results",
                                                  "cartpole_swingup_timeseries.png"))
    args = ap.parse_args()

    scenario = make_cartpole(horizon=args.horizon, n_basis=args.n_basis)
    length, width = scenario.length, float(scenario.width[0])

    if not args.no_train:
        train(args.episodes, args.seed, args.horizon, args.n_basis,
              args.gp_iterations, args.policy_iterations, args.cache)
    control = Controller(args.cache)

    # clean start from the nominal initial state (hanging down, at rest at x=0)
    x0 = np.zeros(4)
    t, states, us, step_costs = simulate(control, x0, args.horizon, scenario.dt,
                                         length, width, args.substeps)
    per_step = step_costs[::args.substeps][:args.horizon]
    total = float(per_step.sum())
    # the same controller on the noisy system, for an honest cost comparison
    rng = np.random.default_rng(0)
    noisy = []
    for _ in range(20):
        xs = x0 + rng.standard_normal(4) * scenario.process_noise
        c = 0.0
        for _ in range(args.horizon):
            c += step_cost(xs, length, width)          # charged before the step
            u = float(np.clip(control(xs), -10.0, 10.0))
            xs = scenario.step_true(xs, u) + rng.standard_normal(4) * scenario.process_noise
        noisy.append(c)

    up = np.degrees(np.abs(states[:, 3] - np.pi))
    # "settled" = the last 40 % of the episode stays within 20 deg of upright
    tail = up[int(0.6 * len(up)):]
    settled = bool(tail.max() < 20.0)
    within = up < 20.0
    first = (float(np.argmax(within)) * scenario.dt / args.substeps
             if within.any() else float("nan"))
    summary = {
        "seed": args.seed,
        "episodes_trained": None if args.no_train
        else int(np.load(args.cache)["costs"].size),
        "learned_episode_costs": [round(float(c), 2)
                                  for c in np.atleast_1d(control.costs)],
        "clean_total_cost": round(total, 3),
        "noisy_total_cost_mean": round(float(np.mean(noisy)), 3),
        "noisy_total_cost_sd": round(float(np.std(noisy)), 3),
        "first_time_within_20deg_s": round(float(first), 3),
        "max_angle_error_tail_deg": round(float(tail.max()), 2),
        "settled_upright": settled,
        "max_abs_cart_x_m": round(float(np.abs(states[:, 0]).max()), 3),
        "max_abs_control_N": round(float(np.abs(us).max()), 2),
    }
    print(json.dumps(summary, indent=1))
    with open(os.path.join(_ROOT, "results", "cartpole_swingup_summary.json"),
              "w") as fh:
        json.dump(summary, fh, indent=1)

    verdict = ("settled upright" if settled else "did NOT settle")
    render(t, states, us, step_costs, np.atleast_1d(control.costs), length, width,
           args.horizon, scenario.dt, args.substeps, args.gif, args.mp4,
           args.fps, verdict)
    timeseries_figure(t, states, us, step_costs, np.atleast_1d(control.costs),
                      length, width, args.horizon, scenario.dt, args.substeps,
                      args.png, args.seed,
                      "settled" if settled else "not settled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
