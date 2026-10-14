"""Run PILCO on cart-pole and report the realised cost per episode.

Reproduces the *faithful* protocol of the NumPy reference (an ``H``-step random
initial rollout, 150 policy optimisations) rather than the crippled 12-step /
10-iteration protocol used by ``pilco_python/scenarios/cartPole/benchmark_de.py``.

Usage::

    python scripts/run_cartpole.py [--episodes 6] [--seed 0] [--kind mean]
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
from pilco.scenarios.cartpole import make_cartpole          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--gp-iterations", type=int, default=150)
    ap.add_argument("--policy-iterations", type=int, default=150)
    ap.add_argument("--n-basis", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default="mean",
                    choices=["mean", "ucb", "gittins"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--info-weight", type=float, default=0.0)
    ap.add_argument("--no-cross-terms", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scenario = make_cartpole(horizon=args.horizon, n_basis=args.n_basis)
    objective = ExplorationObjective(kind=args.kind, beta=args.beta,
                                     info_weight=args.info_weight,
                                     cross_terms=not args.no_cross_terms)
    config = PilcoConfig(horizon=args.horizon, episodes=args.episodes,
                         gp_iterations=args.gp_iterations,
                         policy_iterations=args.policy_iterations,
                         seed=args.seed, objective=objective, verbose=True)

    t0 = time.time()
    model, logs = learn(scenario, config)
    elapsed = time.time() - t0
    costs = [float(np.sum(l.realised_costs)) for l in logs]
    print(f"\n{args.kind.upper()} (info_weight={args.info_weight}) "
          f"seed={args.seed}: total={sum(costs):.2f} costs={[round(c, 2) for c in costs]}"
          f"  [{elapsed:.0f}s]")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"kind": args.kind, "seed": args.seed,
                       "costs": costs, "total": sum(costs),
                       "elapsed": elapsed}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
