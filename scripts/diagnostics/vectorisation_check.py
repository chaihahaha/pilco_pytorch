"""Check that the batched objectives agree with their per-pair loop reference."""
import os, sys
import numpy as np
import torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
torch.set_default_dtype(torch.float64)
from pilco.objectives import expected_kernel_matrix, expected_kernel_matrix_loop  # noqa: E402
from pilco.scenarios.cartpole import make_cartpole                                # noqa: E402
from pilco.trajectory import plan                                                 # noqa: E402
sys.path.insert(0, _HERE)
from profile_hotspots import make_model                                           # noqa: E402

scenario, model = make_model(n_data=60)
traj = plan(model, scenario.optimisation_state(), 8)
means, covs = traj.gp_inputs(scenario.spec)
worst = 0.0
for e in range(model.dynamics.n_out):
    a = expected_kernel_matrix(model.dynamics, means, covs, e)
    b = expected_kernel_matrix_loop(model.dynamics, means, covs, e)
    worst = max(worst, float((a - b).abs().max()))
print(f"expected_kernel_matrix: batched vs loop max abs diff = {worst:.3e}")
print("OK" if worst < 1e-10 else "MISMATCH")
