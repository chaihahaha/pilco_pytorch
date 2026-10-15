"""Targeted follow-up experiments for the closed-loop reproduction question.

Two questions, run sequentially to avoid CPU contention:

**A. Can the PyTorch GP fit be made to give the reference's policy quality?**
   ``gp_fit_effect.py`` showed the unconstrained fit gives policy J 35.80 while
   NumPy's fit gives 33.85, and that the reference's fit is much *smoother*
   (log-lengthscales up to 5.95 vs 2.73).  The ``ls_range`` curb bounds how far
   the lengthscale may drift from the input spread.  This script sweeps it.

**B. Does the step-3 acceptance guard stabilise the closed loop?**
   8 episodes with ``accept_if_improved`` on, everything else identical to the
   recorded run that collapsed after episode 4.

Usage::

    python scripts/diagnostics/followup_experiments.py [--part A|B|both]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
torch.set_default_dtype(torch.float64)

from pilco.models.gp import fit_hyperparameters                          # noqa: E402
from pilco.objectives import ExplorationObjective                        # noqa: E402
from pilco.pilco import PilcoConfig, learn, optimise_policy              # noqa: E402
from pilco.scenarios.cartpole import make_cartpole                       # noqa: E402
from pilco.trajectory import plan                                        # noqa: E402

from gp_fit_effect import build_data, evaluate, make_model               # noqa: E402  (reuse)


def part_a():
    s, dm = build_data()
    H = s["H"]
    rows = []
    for label, kwargs in (
        ("ls_range none (ref fit)", None),
        ("ls_range 4.6 (=PILCO)", {"ls_range": 4.605170185988092}),
        ("ls_range 3.0", {"ls_range": 3.0}),
        ("ls_range 2.0", {"ls_range": 2.0}),
        ("ls_range 1.5", {"ls_range": 1.5}),
    ):
        _, model = make_model(dm)
        if kwargs is None:
            # the reference fit itself: copy numpy's hyper-parameters in
            model.dynamics.set_log_hyp(torch.as_tensor(np.asarray(dm["hyp"])))
            tag = "reference hyp (no fitting)"
        else:
            fit_hyperparameters(model.dynamics, iterations=150, **kwargs)
            tag = label
        val, n, dt, _ = evaluate(model, H)
        ell = model.dynamics.log_lengthscales.detach().numpy()[:5, 0]
        rows.append({"label": tag, "nlml": float(model.dynamics.nlml()),
                     "policy_J": val, "log_ell_out0": [round(float(v), 3) for v in ell]})
        print(f"  {tag:26s} -NLML={float(model.dynamics.nlml()):10.3f} "
              f"policy J={val:.4f}  log_ell[out0]={np.round(ell, 2)}")
    json.dump(rows, open("results/followup_lengthscale_curb.json", "w"), indent=1)
    return rows


def part_b():
    out = []
    for label, accept in (("gating OFF", False), ("gating ON ", True)):
        cfg = PilcoConfig(horizon=40, episodes=8, gp_iterations=30,
                          policy_iterations=60, seed=0, verbose=False,
                          accept_if_improved=accept, keep_best_hyperparameters=False,
                          objective=ExplorationObjective(kind="mean"))
        t0 = time.time()
        model, logs = learn(make_cartpole(), cfg, log=None)
        costs = [round(float(np.sum(l.realised_costs)), 1) for l in logs]
        acc = [int(l.accepted) for l in logs]
        out.append({"label": label, "costs": costs, "accepted": acc,
                    "seconds": round(time.time() - t0, 1)})
        print(f"  {label}: costs={costs}")
        print(f"  {' ' * len(label)}  accepted={acc}  [{time.time() - t0:.0f}s]")
    json.dump(out, open("results/followup_gating.json", "w"), indent=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=["A", "B", "both"])
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if args.part in ("A", "both"):
        print("A) lengthscale-curb sweep (same 40-point dataset, same optimiser)")
        part_a()
    if args.part in ("B", "both"):
        print("\nB) step-3 acceptance guard, 8 episodes, gp=30 policy=60")
        part_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
