"""Benchmark PILCO against the closed-form directed-exploration objectives.

Everything runs through the same policy optimiser (L-BFGS on the closed-form
objective), so the only difference between the rows is the objective:

* ``pilco``    -- expected cumulative cost
* ``ucb``      -- ``E[C] - beta * sd[C]`` with the analytic cumulative variance
* ``gittins``  -- ``E[C] + lambda'_gamma * sd[C]``
* ``ig``       -- ``E[C] - w * information_gain`` (closed-form GP mutual
  information; no cost-variance term)
* ``ucb+ig``   -- both

Usage::

    python scripts/benchmark_exploration.py --env cartpole --episodes 5 \
        --seeds 0 1 2 --methods pilco ucb gittins ig
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

from pilco.objectives import ExplorationObjective          # noqa: E402
from pilco.pilco import PilcoConfig, learn                  # noqa: E402

METHODS = {
    "pilco": lambda a: ExplorationObjective(kind="mean"),
    "ucb": lambda a: ExplorationObjective(kind="ucb", beta=a.beta),
    "gittins": lambda a: ExplorationObjective(kind="gittins"),
    "ig": lambda a: ExplorationObjective(kind="mean", info_weight=a.info_weight),
    "ucb+ig": lambda a: ExplorationObjective(kind="ucb", beta=a.beta,
                                             info_weight=a.info_weight),
}


def make_scenario(name: str, horizon: int, n_basis: int):
    if name == "cartpole":
        from pilco.scenarios.cartpole import make_cartpole
        return make_cartpole(horizon=horizon, n_basis=n_basis)
    if name == "mountaincar":
        from pilco.scenarios.mountaincar import make_mountaincar
        return make_mountaincar(horizon=horizon, n_basis=n_basis)
    raise ValueError(f"unknown environment {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cartpole", choices=["cartpole", "mountaincar"])
    ap.add_argument("--methods", nargs="+", default=["pilco", "ucb", "gittins", "ig"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--gp-iterations", type=int, default=150)
    ap.add_argument("--policy-iterations", type=int, default=150)
    ap.add_argument("--n-basis", type=int, default=12)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--info-weight", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    default_h = 40 if args.env == "cartpole" else 120
    horizon = args.horizon or default_h

    results: dict = {}
    for method in args.methods:
        rows = []
        t0 = time.time()
        for seed in args.seeds:
            scenario = make_scenario(args.env, horizon, args.n_basis)
            objective = METHODS[method](args)
            config = PilcoConfig(horizon=horizon, episodes=args.episodes,
                                 gp_iterations=args.gp_iterations,
                                 policy_iterations=args.policy_iterations,
                                 seed=seed, objective=objective, verbose=False)
            _, logs = learn(scenario, config, log=None)
            costs = [float(np.sum(l.realised_costs)) for l in logs]
            rows.append({"seed": seed, "costs": costs, "total": sum(costs)})
            print(f"  {method:8s} seed={seed}: total={sum(costs):8.2f} "
                  f"{[round(c, 1) for c in costs]}")
        results[method] = {"rows": rows,
                           "mean": float(np.mean([r["total"] for r in rows])),
                           "elapsed": time.time() - t0}

    base = results.get("pilco", {}).get("mean")
    print(f"\n{'=' * 66}")
    print(f"{args.env}: {args.episodes} episodes, horizon {horizon}, "
          f"seeds {args.seeds}")
    print(f"{'method':>10} {'mean total':>12} {'vs PILCO':>10}  per-seed totals")
    print("-" * 66)
    for method in args.methods:
        r = results[method]
        totals = " ".join(f"{row['total']:.0f}" for row in r["rows"])
        pct = "" if base is None or method == "pilco" \
            else f"{100 * (r['mean'] - base) / abs(base):+.1f}%"
        print(f"{method:>10} {r['mean']:>12.1f} {pct:>10}  {totals}")
    print("=" * 66)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "results": results}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
