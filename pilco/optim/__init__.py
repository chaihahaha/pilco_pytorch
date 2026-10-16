"""Optimisers: a faithful PyTorch port of PILCO's MATLAB-style ``minimize``."""
from .minimize import MinimizeResult, ParamVector, autograd_objective, minimize

__all__ = ["minimize", "MinimizeResult", "ParamVector", "autograd_objective"]
