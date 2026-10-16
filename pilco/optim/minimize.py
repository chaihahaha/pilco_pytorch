"""A faithful PyTorch port of PILCO's MATLAB-style ``minimize`` (BFGS + its own
line search).

Why this exists
---------------
The closed-loop gap between ``pilco_py`` and ``pilco_python`` was traced
(``scripts/diagnostics/reproduction_check.py``) to the point where the GP fit
stops: NumPy's ``minimize`` halts at a smoother, slightly under-fitted GP than
``torch.optim.LBFGS`` (strong Wolfe) does, and at 40 data points that happens to
control better.  Rather than argue about which optimiser is "better", the honest
move is to make the optimiser a controlled variable -- hence this port.

The algorithm is transcribed from ``pilco_python/util/minimize.py``, which is
itself a line-by-line translation of Carl Rasmussen's ``minimize.m``:

* ``length > 0`` counts **line searches** (each allowed up to ``MFEPLS`` function
  evaluations); ``length < 0`` counts function evaluations.  This distinction is
  exactly what made ``torch.optim.LBFGS(max_iter=150)`` (roughly 187 function
  evaluations) incomparable with ``minimize(length=150, MFEPLS=30)``.
* BFGS with the same rank-2 update, ``SIG = 0.5``, ``MSR = 100``.
* The same cubic-extrapolation / cubic-interpolation line search with Wolfe-
  Powell checks (``_minCubic``, ``_wp_*``).

Integration
-----------
The objective is supplied as ``evaluate(x_flat) -> (float, Tensor)``.  For
autograd-differentiable objectives the wrapper :class:`ParamVector` writes the
flat vector into an ``nn.Parameter`` list under ``no_grad`` and then reads the
gradient back with ``torch.autograd.grad``, which makes the flat vector behave as
if the objective were a pure function of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import torch

__all__ = ["minimize", "MinimizeResult", "ParamVector", "flatten_params"]

_INT = 0.1   # interpolation limit for the cubic step
_EXT = 5.0   # extrapolation limit for the cubic step


class ParamVector:
    """A flat ``torch.Tensor`` view of a list of ``nn.Parameter``s.

    ``set`` writes the vector into the parameters (under ``no_grad``) and
    ``grad`` reads their gradients back out, so that an objective built on top of
    ``nn.Module``s can be driven by :func:`minimize` as if it were a function of
    the flat vector alone.
    """

    def __init__(self, params: Sequence[torch.Tensor]):
        self.params = list(params)
        self.sizes = [p.numel() for p in self.params]
        self.shapes = [p.shape for p in self.params]

    @property
    def numel(self) -> int:
        return sum(self.sizes)

    def flat(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    def set(self, x: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for p, n, shape in zip(self.params, self.sizes, self.shapes):
                p.copy_(x[offset:offset + n].reshape(shape))
                offset += n

    def grad(self) -> Optional[torch.Tensor]:
        if any(p.grad is None for p in self.params):
            return None
        return torch.cat([p.grad.reshape(-1) for p in self.params])

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def flatten_params(params: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def autograd_objective(fn: Callable[[], torch.Tensor],
                       vector: ParamVector) -> Callable[[torch.Tensor], tuple]:
    """Wrap a scalar-valued ``fn()`` as ``evaluate(x_flat) -> (value, grad)``.

    ``fn`` must depend on ``vector``'s parameters; it is re-evaluated at every
    query point with the parameters set from ``x_flat``.
    """

    def evaluate(x: torch.Tensor):
        vector.set(x)
        value = fn()
        grads = torch.autograd.grad(value, vector.params, allow_unused=False)
        return float(value.detach()), torch.cat([g.detach().reshape(-1) for g in grads])

    return evaluate


@dataclass
class MinimizeResult:
    x: torch.Tensor
    fX: list[float]
    iterations: int
    info: dict = field(default_factory=dict)

    @property
    def final(self) -> float:
        return self.fX[-1]


def minimize(x0: torch.Tensor, evaluate: Callable[[torch.Tensor], tuple],
             length: int = 150, MFEPLS: int = 30, method: str = "BFGS",
             SIG: Optional[float] = None, MSR: float = 100.0,
             verbosity: int = 0) -> MinimizeResult:
    """Minimise ``evaluate`` starting from ``x0``.

    Parameters
    ----------
    x0:
        Flat starting point (a 1-D ``float64`` tensor).
    evaluate:
        ``evaluate(x) -> (float, Tensor)`` returning the value and the gradient
        at ``x``.
    length:
        Number of line searches (> 0) or function evaluations (< 0), matching
        PILCO's ``minimize``.
    MFEPLS:
        Maximum function evaluations per line search.
    method:
        ``"BFGS"`` (default), ``"CG"`` or ``"LBFGS"``.  Only BFGS is exercised by
        the learning loop; the others are included for completeness.
    """
    x = x0.detach().clone()
    fx0, dfx0 = evaluate(x)
    if not (math.isfinite(fx0) and bool(torch.isfinite(dfx0).all())):
        raise ValueError("minimize: initial value or gradient is not finite")

    if method == "BFGS":
        return _bfgs(x, fx0, dfx0, evaluate, length, MFEPLS, SIG or 0.5, MSR, verbosity)
    if method == "CG":
        return _cg(x, fx0, dfx0, evaluate, length, MFEPLS, SIG or 0.1, MSR, verbosity)
    if method == "LBFGS":
        return _lbfgs(x, fx0, dfx0, evaluate, length, MFEPLS, SIG or 0.5, MSR, verbosity)
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# Wolfe-Powell conditions (PILCO's wp())
# ---------------------------------------------------------------------------
class _WolfePowell:
    def __init__(self, sig: float, rho: float = 0.0):
        self.sig = sig
        self.rho = rho
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0

    def setup(self, point: dict) -> None:
        self.a = self.rho * point["s"]
        self.b = point["f"]
        self.c = -self.sig * point["s"]

    def check(self, point: dict) -> int:
        if point["f"] > self.a * point["x"] + self.b:
            return -1 if self.a > 0 else -2
        if point["s"] < -self.c:
            return 0
        if point["s"] > self.c:
            return 1
        return 2


def _min_cubic(x: float, df: float, s0: float, s1: float, extr: bool) -> float:
    """Minimiser of the approximating cubic (PILCO's ``minCubic``)."""
    A = -6 * df + 3 * (s0 + s1) * x
    B = 3 * df - (2 * s0 + s1) * x
    disc = B * B - A * s0 * x
    if disc < 0:
        disc = 0.0
    denom = B + math.sqrt(disc)
    if abs(denom) < 1e-16:
        z = x / 2
    else:
        z = -s0 * x * x / denom
    if extr:
        if (not math.isfinite(z)) or z < x or z > x * _EXT:
            z = _EXT * x
        z = max(z, (1 + _INT) * x)
    else:
        if (not math.isfinite(z)) or z < 0 or z > x:
            z = x / 2
        z = min(max(z, _INT * x), (1 - _INT) * x)
    return z


def _line_search(x0: torch.Tensor, f0: float, df0: torch.Tensor, d: torch.Tensor,
                 s: float, a: float, i: int, evaluate, length: int,
                 MFEPLS: int, SIG: float, verbosity: int):
    """PILCO's ``lineSearch``: cubic extrapolation then cubic interpolation."""
    limit = min(MFEPLS, -i - length) if length < 0 else MFEPLS
    wp = _WolfePowell(SIG, 0.0)

    p0 = {"x": 0.0, "f": f0, "df": df0, "s": s}
    p1 = dict(p0)
    j = 0
    p3 = {"x": a}
    wp.setup(p0)

    # ---- extrapolation ---------------------------------------------------
    while True:
        ok = False
        while not ok and j < limit:
            try:
                j += 1
                new_x = x0 + p3["x"] * d
                p3["f"], p3["df"] = evaluate(new_x)
                p3["s"] = float(p3["df"] @ d)
                ok = True
                if not (math.isfinite(p3["f"]) and math.isfinite(p3["s"])):
                    raise ValueError("objective returned Inf or NaN")
            except Exception:
                p3["x"] = (p1["x"] + p3["x"]) / 2
                ok = False
                p3["f"] = float("nan")
                p3["s"] = float("nan")
        if wp.check(p3) or j >= limit:
            break
        p0 = dict(p1)
        p1 = dict(p3)
        dx = p1["x"] - p0["x"]
        dfdiff = p1["f"] - p0["f"]
        p3["x"] = p0["x"] + _min_cubic(dx, dfdiff, p0["s"], p1["s"], True)

    # ---- interpolation ---------------------------------------------------
    while True:
        if not (math.isfinite(p3["f"]) and math.isfinite(p3["s"])):
            p2 = dict(p1)
            break
        p2 = dict(p3) if p1["f"] > p3["f"] else dict(p1)
        if wp.check(p2) > 1 or j >= limit:
            break

        dx = p3["x"] - p1["x"]
        dfdiff = p3["f"] - p1["f"]
        p2["x"] = p1["x"] + _min_cubic(dx, dfdiff, p1["s"], p3["s"], False)

        ok = False
        while not ok and j < limit:
            try:
                j += 1
                new_x = x0 + p2["x"] * d
                p2["f"], p2["df"] = evaluate(new_x)
                p2["s"] = float(p2["df"] @ d)
                ok = True
                if not (math.isfinite(p2["f"]) and math.isfinite(p2["s"])):
                    raise ValueError("objective returned Inf or NaN")
            except Exception:
                p2["x"] = (p1["x"] + p2["x"]) / 2
                ok = False
                if j >= limit:
                    p2 = dict(p1)

        status = wp.check(p2)
        if (status > -1 and p2["s"] > 0) or status < -1:
            p3 = dict(p2)
        else:
            p1 = dict(p2)

    x_ret = x0 + p2["x"] * d
    i = i + j if length < 0 else i + 1
    if verbosity:
        print(f"linesearch # {i:6d};  value {p2['f']:4.6e}")
    if wp.check(p2) < 2:
        i = -i                       # signal line-search failure
    return x_ret, p2["x"], p2["f"], p2["df"], i


# ---------------------------------------------------------------------------
# BFGS (the method the learning loop uses)
# ---------------------------------------------------------------------------
def _bfgs(x0, fx0, dfx0, evaluate, length, MFEPLS, SIG, MSR, verbosity):
    n = x0.numel()
    i = int(length < 0)
    ok = True
    x = x0.detach().clone()
    fx = [fx0]
    r = -dfx0
    s = -float(r @ r)
    b_step = -1.0 / (s - 1)
    H = torch.eye(n, dtype=x.dtype)
    dfx = dfx0.clone()

    while i < abs(length):
        r_norm = float(torch.linalg.norm(r))
        b_step = min(b_step, 1.0 / max(r_norm, 1e-16))
        b_step = max(b_step, 1e-7 / max(r_norm, 1e-16))
        x, b_step, fx0, dfx, i = _line_search(x, fx0, dfx, r, s, b_step, i,
                                              evaluate, length, MFEPLS, SIG,
                                              verbosity)
        if i < 0:
            i = -i
            if ok:
                ok = False
            else:
                break
        else:
            ok = True
            t = x - x0
            y = dfx - dfx0
            ty = float(t @ y)
            Hy = H @ y
            if abs(ty) > 1e-16:
                yHy = float(y @ Hy)
                H = (H + (ty + yHy) / (ty * ty) * torch.outer(t, t)
                     - (1.0 / ty) * torch.outer(Hy, t)
                     - (1.0 / ty) * torch.outer(t, Hy))
        r = -H @ dfx
        s = float(r @ dfx)
        x0 = x.clone()
        dfx0 = dfx.clone()
        fx.append(fx0)

    return MinimizeResult(x=x, fX=fx, iterations=i,
                          info={"method": "BFGS", "H": H})


# ---------------------------------------------------------------------------
# CG and LBFGS (ported for completeness; not used by the learning loop)
# ---------------------------------------------------------------------------
def _cg(x0, fx0, dfx0, evaluate, length, MFEPLS, SIG, MSR, verbosity):
    i = int(length < 0)
    ok = True
    r = -dfx0
    s = -float(r @ r)
    b_step = -1.0 / (s - 1)
    bs = -1.0
    fx = [fx0]
    x = x0.detach().clone()
    dfx = dfx0.clone()

    while i < abs(length):
        b_step = b_step * bs / min(b_step * s, bs / MSR)
        r_norm = float(torch.linalg.norm(r))
        b_step = min(b_step, 1.0 / max(r_norm, 1e-16))
        b_step = max(b_step, 1e-7 / max(r_norm, 1e-16))
        x, b_step, fx0, dfx, i = _line_search(x, fx0, dfx, r, s, b_step, i,
                                              evaluate, length, MFEPLS, SIG,
                                              verbosity)
        if i < 0:
            i = -i
            if ok:
                ok = False
                r = -dfx
            else:
                break
        else:
            ok = True
            bs = b_step * s
            num = float(dfx @ (dfx - dfx0))
            den = float(dfx0 @ dfx0)
            r = (num / den * r - dfx) if den != 0 else -dfx
        s = float(r @ dfx)
        if s >= 0:
            r = -dfx
            s = float(r @ dfx)
            ok = False
        x0 = x.clone()
        dfx0 = dfx.clone()
        fx.append(fx0)

    return MinimizeResult(x=x, fX=fx, iterations=i, info={"method": "CG"})


def _lbfgs(x0, fx0, dfx0, evaluate, length, MFEPLS, SIG, MSR, verbosity,
           memory: Optional[int] = None):
    n = x0.numel()
    k = 0
    ok = True
    x = x0.detach().clone()
    fx = [fx0]
    bs = -1.0 / MSR
    m = memory or min(100, n)
    a_mem = torch.zeros(m)
    t_mem = torch.zeros(n, m, dtype=x.dtype)
    y_mem = torch.zeros(n, m, dtype=x.dtype)
    rho_mem = torch.zeros(m)
    i = int(length < 0)
    dfx = dfx0.clone()

    while i < abs(length):
        q = dfx.clone()
        for jj in range(min(k, m)):
            j = (k - 1 - jj) % m
            a_mem[j] = float(t_mem[:, j] @ q) / max(float(rho_mem[j]), 1e-16)
            q = q - a_mem[j] * y_mem[:, j]
        if k == 0:
            qn = float(q @ q)
            r = -q / max(qn, 1e-16)
        else:
            j = (k - 1) % m
            denom = float(y_mem[:, j] @ y_mem[:, j])
            r = -float(t_mem[:, j] @ y_mem[:, j]) / max(denom, 1e-16) * q
        for jj in range(min(k, m)):
            j = (k - min(k, m) + jj) % m
            beta = float(y_mem[:, j] @ r) / max(float(rho_mem[j]), 1e-16)
            r = r - t_mem[:, j] * (a_mem[j] + beta)
        s = float(r @ dfx)
        if s >= 0:
            r = -dfx
            s = float(r @ dfx)
            k = 0
            ok = False
        b_step = bs / min(bs, s / MSR)
        r_norm = float(torch.linalg.norm(r))
        b_step = min(b_step, 1.0 / max(r_norm, 1e-16))
        b_step = max(b_step, 1e-7 / max(r_norm, 1e-16))
        if not bool(torch.isfinite(r).all()):
            i = -i
        else:
            x, b_step, fx0, dfx, i = _line_search(x, fx0, dfx, r, s, b_step, i,
                                                  evaluate, length, MFEPLS, SIG,
                                                  verbosity)
        if i < 0:
            i = -i
            if ok:
                ok = False
                k = 0
            else:
                break
        else:
            j = k % m
            t_mem[:, j] = x - x0
            y_mem[:, j] = dfx - dfx0
            rho_mem[j] = float(t_mem[:, j] @ y_mem[:, j])
            ok = True
            k += 1
            bs = b_step * s
        x0 = x.clone()
        dfx0 = dfx.clone()
        fx.append(fx0)

    return MinimizeResult(x=x, fX=fx, iterations=i, info={"method": "LBFGS"})
