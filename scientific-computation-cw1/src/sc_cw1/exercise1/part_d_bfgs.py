"""1(d): inverse-Hessian BFGS; no Hessian evaluation or linear solve."""

import numpy as np
from sc_cw1.common.utilities import (
    NUMERICAL_ERRORS, SolverResult, armijo, checked_evaluation, finite, vector,
    validate_controls, output_dirs, save_history, summary, write_csv, module_cli,
)
from .model import objective, gradient


def inverse_bfgs_update(B, s, y, curvature_rtol=1e-12):
    """Implement the brief's expanded inverse update, preserving product order.

    B+ = B + ((s.T y + y.T B y)/(s.T y)^2) s s.T
             - (B y s.T + s y.T B)/(s.T y).
    Require s.T y > curvature_rtol*||s||*||y|| and >0 before division.
    This is quasi-Newton: secant information replaces second derivatives.
    Raises FloatingPointError on unusable curvature; never fabricates y or B.
    """
    if not np.isfinite(curvature_rtol) or not 0 <= curvature_rtol < 1:
        raise ValueError("curvature_rtol must be in [0,1)")
    B, s, y = np.asarray(B, dtype=float), vector(s), vector(y)
    if B.shape != (2, 2):
        raise ValueError("B must have shape (2,2)")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        finite(B)
        sy = float(finite(s @ y))
        scale = float(finite(np.linalg.norm(s) * np.linalg.norm(y)))
        if sy <= 0 or sy <= curvature_rtol * scale:
            raise FloatingPointError("nonpositive or numerically unsafe BFGS curvature")
        By = B @ y
        # Outer products are essential: no elementwise stand-in for matrix products.
        updated = B + ((sy + y @ By) / sy**2) * np.outer(s, s)
        updated -= (np.outer(By, s) + np.outer(s, y @ B)) / sy
    return finite(updated)


def bfgs(x0, *, fun=objective, grad=gradient, omega=1.0, tol=1e-6, maxit=200,
         step_mode="armijo", alpha=1.0, curvature_action="fail", curvature_rtol=1e-12,
         c1=1e-4, shrink=0.5, max_backtracks=50):
    """Use p=-B∇J, s=alpha*p, y=∇J(x+s)-∇J(x), and the printed B update.

    B0=omega I, omega>0. Stop strictly on ||∇J||_2<tol.
    Invalid curvature either fails explicitly (default), or retains B with a
    recorded 'skipped_curvature' action when requested. No hidden reset, damping,
    Hessian call or linear solve. The fixed-step comparison uses action='fail'.
    """
    validate_controls(tol, maxit)
    x = vector(x0)
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("omega and alpha must be finite and positive")
    if step_mode not in ("fixed", "armijo") or curvature_action not in ("fail", "skip"):
        raise ValueError("invalid step mode or curvature action")
    if not (0 <= curvature_rtol < 1 and 0 < c1 < 1 and 0 < shrink < 1):
        raise ValueError("invalid curvature or line-search controls")
    validate_controls(tol, max_backtracks)
    B = omega * np.eye(2)
    result = SolverResult(parameters=dict(tol=tol, maxit=maxit, omega=omega,
        step_mode=step_mode, initial_alpha=alpha, curvature_action=curvature_action,
        curvature_rtol=curvature_rtol, c1=c1, shrink=shrink, max_backtracks=max_backtracks,
        stopping_statistic="gradient_norm"))
    try:
        f, g, gn = checked_evaluation(fun, grad, x)
    except NUMERICAL_ERRORS as exc:
        return result.stop("numerical_failure", f"initial evaluation: {exc}")
    result.append(x, f, gn, step_kind="initial", update_action="initial")
    if gn < tol:
        return result.stop("converged", "initial gradient norm < tolerance")
    for _ in range(maxit):
        if gn < tol:
            return result.stop("converged", "gradient norm < tolerance")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                p = finite(-B @ g)
                slope = float(finite(g @ p))
            a = alpha if step_mode == "fixed" else armijo(
                fun, x, f, g, p, initial_alpha=alpha, c1=c1,
                shrink=shrink, max_backtracks=max_backtracks)
            if a is None:
                return result.stop("line_search_failure", "non-descent direction or Armijo exhausted")
            with np.errstate(over="raise", invalid="raise"):
                s = finite(a*p)
                xn = finite(x + s)
            fn, gnew, gnnew = checked_evaluation(fun, grad, xn)
            if np.array_equal(xn, x):
                return result.stop("numerical_failure", "floating-point stagnation above tolerance")
            with np.errstate(over="raise", invalid="raise"):
                y = finite(gnew - g)
                sy = float(finite(s @ y))
        except NUMERICAL_ERRORS as exc:
            return result.stop("numerical_failure", str(exc))
        result.append(xn, fn, gnnew, a, step_kind="bfgs", direction1=float(p[0]),
            direction2=float(p[1]), directional_derivative=slope, curvature=sy,
            update_action="not_needed_converged" if gnnew < tol else "pending")
        x, f, g, gn = xn, fn, gnew, gnnew
        if gn < tol:
            return result.stop("converged", "gradient norm < tolerance")
        try:
            B = inverse_bfgs_update(B, s, y, curvature_rtol)
            result.history[-1]["update_action"] = "inverse_bfgs"
        except NUMERICAL_ERRORS as exc:
            result.history[-1]["update_action"] = "failed_curvature"
            if curvature_action == "fail":
                return result.stop("numerical_failure", str(exc))
            result.history[-1]["update_action"] = "skipped_curvature"
    return result


def run(output="results"):
    """Run standalone inverse BFGS with Armijo from (-2,1), omega=0.01."""
    root = output_dirs(output)
    result = bfgs((-2, 1), omega=0.01)
    save_history(root / "tables/ex1_part_d_bfgs_history.csv", result)
    write_csv(root / "tables/ex1_part_d_summary.csv", [summary(result)])
    return result


if __name__ == "__main__":
    module_cli(run)
