"""pilco: a PyTorch re-implementation of PILCO with modular moment-matching operators.

See ``docs/ARCHITECTURE.md`` for the design rationale.
"""
from .gaussian import Gaussian
from .ops.base import MomentOp, Moments, splice
from .ops.fourier import TrigAugmentOp, SaturationOp, FourierOp
from .ops.policy import RbfPolicyOp
from .ops.gp import GpDynamicsOp
from .ops.cost import SaturatingCostOp

__all__ = [
    "Gaussian", "MomentOp", "Moments", "splice",
    "TrigAugmentOp", "SaturationOp", "FourierOp",
    "RbfPolicyOp", "GpDynamicsOp", "SaturatingCostOp",
]
