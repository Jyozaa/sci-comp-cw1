"""Small shared helpers; the algorithm updates remain in each question's file."""

from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import Callable

import numpy as np

NUMERICAL_ERRORS = (FloatingPointError, OverflowError, np.linalg.LinAlgError)


def vector(x):
    """Copy a length-two real vector; nonfinite entries are evaluated by solvers."""
    a = np.asarray(x, dtype=float)
    if a.shape != (2,):
        raise ValueError("Expected a two-dimensional vector of shape (2,).")
    return a.copy()


def validate_controls(tol, maxit):
    """Require positive finite tolerance and a nonnegative integer step budget."""
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if isinstance(maxit, bool) or not isinstance(maxit, (int, np.integer)) or maxit < 0:
        raise ValueError("maxit must be a nonnegative integer")


def finite(value):
    """Raise on NaN/Inf so they cannot be mistaken for convergence."""
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("nonfinite numerical value")
    return value


def checked_evaluation(fun, grad, x):
    """Evaluate a scalar function and its length-two gradient under strict FP checks."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        finite(x)
        f = float(finite(fun(x)))
        g = vector(finite(grad(x)))
        n = float(finite(np.linalg.norm(g)))
    return f, g, n


@dataclass
class SolverResult:
    """History contains x_0 and all successfully evaluated iterates.

    Each row's alpha/direction describes the step *into* that row; row zero
    has alpha=0. Failed trial points are excluded and explained in message.
    nit counts completed finite updates, never attempted or rejected steps.
    """

    history: list[dict] = field(default_factory=list)
    status: str = "max_iterations"
    message: str = "iteration budget exhausted"
    parameters: dict = field(default_factory=dict)

    @property
    def converged(self):
        """True only when the specified stopping statistic satisfies tolerance."""
        return self.status == "converged"

    @property
    def nit(self):
        """Number of completed updates (zero when initial evaluation failed)."""
        return max(0, len(self.history) - 1)

    @property
    def x(self):
        """Last valid point, or NaNs if the initial evaluation failed."""
        return self.iterates[-1].copy() if self.history else np.full(2, np.nan)

    @property
    def iterates(self):
        """Array with shape (nit+1, 2), or (0, 2) on initial failure."""
        return np.array([[r["x1"], r["x2"]] for r in self.history]).reshape(-1, 2)

    def stop(self, status, message):
        """Set an explicit exit reason and return this result."""
        self.status, self.message = status, message
        return self

    def append(self, x, value, gradient_norm, alpha=0.0, **details):
        """Append a finite iterate and scalar diagnostics, copying its coordinates."""
        self.history.append(dict(k=len(self.history), x1=float(x[0]), x2=float(x[1]),
                                 value=float(value), gradient_norm=float(gradient_norm),
                                 alpha=float(alpha), **details))


def finite_difference_jacobian(fun: Callable, x, h=1e-5):
    """Independent central differences: column j=(f(x+h_j e_j)-f(x-h_j e_j))/(2h_j).

    Handles either scalar or vector functions. h_j=h*max(1,|x_j|).
    A scalar function gives a 1-by-2 array; no analytical derivatives are used.
    """
    x = vector(x)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    cols = []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        finite(x)
        for j in range(2):
            step = h * max(1.0, abs(x[j]))
            e = np.zeros(2)
            e[j] = step
            cols.append(np.atleast_1d((fun(x + e) - fun(x - e)) / (2 * step)))
    return finite(np.column_stack(cols))


def armijo(fun, x, value, gradient, direction, *, c1=1e-4, shrink=0.5,
           initial_alpha=1.0, max_backtracks=50):
    """Return alpha satisfying f(x+alpha*p)<=f(x)+c1*alpha*g.T*p, else None.

    Objective Armijo adapts Unit 9's residual-based step-halving to optimisation.
    Requires g.T*p<0. Overflowed trials are rejected. No relaxed inequality or
    forced acceptance is used; initial trial plus max_backtracks retries.
    """
    if not (0 < c1 < 1 and 0 < shrink < 1 and np.isfinite(initial_alpha)
            and initial_alpha > 0):
        raise ValueError("invalid Armijo parameters")
    validate_controls(1.0, max_backtracks)
    with np.errstate(over="raise", invalid="raise"):
        slope = float(finite(gradient @ direction))
    if slope >= 0:
        return None
    alpha = initial_alpha
    for _ in range(max_backtracks + 1):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                trial = finite(x + alpha * direction)
                candidate = float(finite(fun(trial)))
                bound = float(finite(value + c1 * alpha * slope))
            if candidate <= bound:
                return alpha
        except NUMERICAL_ERRORS:
            pass
        alpha *= shrink
    return None


def write_csv(path, rows, fieldnames=None):
    """Write dictionaries with stable column ordering and actual floating values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_history(path, result):
    """Export iterate diagnostics plus final status and stopping criterion per row."""
    write_csv(path, [dict(row, final_status=result.status,
                          stopping_statistic=result.parameters["stopping_statistic"],
                          tolerance=result.parameters["tol"]) for row in result.history])


def summary(result, **labels):
    """Compact run summary; no successful history means undefined final metrics."""
    row = result.history[-1] if result.history else {}
    return dict(labels, status=result.status, converged=result.converged, iterations=result.nit,
                x1=float(result.x[0]), x2=float(result.x[1]),
                value=row.get("value", np.nan), gradient_norm=row.get("gradient_norm", np.nan),
                residual_norm=row.get("residual_norm", np.nan), message=result.message,
                **result.parameters)


def output_dirs(output):
    """Create results/{figures,tables} under a caller-selected results directory."""
    root = Path(output)
    for name in ("figures", "tables"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def pyplot():
    """Load Matplotlib with a noninteractive backend, independent of display state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, path):
    """Export a labelled Matplotlib figure at 300 DPI and release its resources."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    pyplot().close(fig)


def plot_path(ax, result, label, color="C0", marker="o"):
    """Plot all iterates and distinguish start, intermediates and final point."""
    p = result.iterates
    if len(p):
        ax.plot(p[:, 0], p[:, 1], color=color, marker=marker, markersize=2.3,
                linewidth=1.1, label=label)
        ax.scatter(*p[0], c="black", marker="s", s=45, zorder=5, label="Start")
        ax.scatter(*p[-1], c=color, marker="*", s=130, edgecolors="black", zorder=6,
                   label="Final iterate")


def module_cli(run):
    """Standard --output interface for independently runnable question modules."""
    import argparse
    parser = argparse.ArgumentParser(description=run.__doc__)
    parser.add_argument("--output", default="results", help="results directory")
    args = parser.parse_args()
    run(args.output)
