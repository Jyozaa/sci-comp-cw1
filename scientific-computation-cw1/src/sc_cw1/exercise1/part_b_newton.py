"""1(b): Newton optimisation, with explicitly selected line search/safeguards."""

import numpy as np
from sc_cw1.common.utilities import (
    NUMERICAL_ERRORS, SolverResult, armijo, checked_evaluation, finite, vector,
    validate_controls, output_dirs, save_history, summary, write_csv, module_cli,
)
from .model import objective, gradient, hessian


def newton(x0, *, fun=objective, grad=gradient, hess=hessian, tol=1e-6, maxit=200,
           step_mode="armijo", alpha=1.0, fallback="none", condition_limit=1e12,
           c1=1e-4, shrink=0.5, max_backtracks=50):
    """Solve H(x_k) p_k=-∇J(x_k), then x_{k+1}=x_k+alpha_k p_k.

    Stop strictly on ||∇J||_2<tol; maxit counts updates, excluding x0.
    step_mode='fixed' uses exactly alpha; 'armijo' uses objective decrease.
    fallback='steepest_descent' explicitly replaces singular/ill-conditioned or
    non-descent Armijo directions with -g. 'none' returns failure instead.
    Fixed mode accepts non-descent Newton steps (stationarity solving), and
    never silently enables safeguards. Every accepted direction is recorded.
    Custom callables allow validation on known quadratics.
    """
    validate_controls(tol, maxit)
    x = vector(x0)
    if step_mode not in ("fixed", "armijo") or fallback not in ("none", "steepest_descent"):
        raise ValueError("invalid step mode or fallback")
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("alpha>0 and condition_limit>1 must be finite")
    if not (0 < c1 < 1 and 0 < shrink < 1):
        raise ValueError("invalid line-search constants")
    validate_controls(tol, max_backtracks)
    result = SolverResult(parameters=dict(tol=tol, maxit=maxit, step_mode=step_mode,
        initial_alpha=alpha, fallback=fallback, condition_limit=condition_limit,
        c1=c1, shrink=shrink, max_backtracks=max_backtracks, stopping_statistic="gradient_norm"))
    try:
        f, g, gn = checked_evaluation(fun, grad, x)
    except NUMERICAL_ERRORS as exc:
        return result.stop("numerical_failure", f"initial evaluation: {exc}")
    result.append(x, f, gn, step_kind="initial")
    for _ in range(maxit):
        if gn < tol:
            return result.stop("converged", "gradient norm < tolerance")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                H = np.asarray(finite(hess(x)), dtype=float)
                if H.shape != (2, 2):
                    raise ValueError("Hessian must have shape (2,2)")
                condition = float(np.linalg.cond(H))
                kind = "newton"
                if not np.isfinite(condition) or condition > condition_limit:
                    if fallback == "none":
                        return result.stop("numerical_failure", "singular or ill-conditioned Hessian")
                    p, kind = -g, "fallback_condition"
                else:
                    try:
                        p = finite(np.linalg.solve(H, -g))
                    except np.linalg.LinAlgError:
                        if fallback == "none":
                            raise
                        p, kind = -g, "fallback_singular"
                slope = float(finite(g @ p))
                if step_mode == "armijo" and slope >= 0:
                    if fallback == "none":
                        return result.stop("line_search_failure", "Newton direction is not descent")
                    p, kind = -g, "fallback_non_descent"
                    slope = float(g @ p)
            a = alpha if step_mode == "fixed" else armijo(
                fun, x, f, g, p, initial_alpha=alpha, c1=c1,
                shrink=shrink, max_backtracks=max_backtracks)
            if a is None:
                return result.stop("line_search_failure", "Armijo backtracking exhausted")
            with np.errstate(over="raise", invalid="raise"):
                xn = finite(x + a*p)
            fn, gnew, gnnew = checked_evaluation(fun, grad, xn)
            if np.array_equal(xn, x):
                return result.stop("numerical_failure", "floating-point stagnation above tolerance")
        except NUMERICAL_ERRORS as exc:
            return result.stop("numerical_failure", str(exc))
        result.append(xn, fn, gnnew, a, step_kind=kind, direction1=float(p[0]),
                      direction2=float(p[1]), directional_derivative=slope, hessian_condition=condition)
        x, f, g, gn = xn, fn, gnew, gnnew
    if gn < tol:
        return result.stop("converged", "gradient norm < tolerance at final allowed update")
    return result


def run(output="results"):
    """Run 1(b) from four illustrative starts with explicit descent fallback."""
    root = output_dirs(output)
    runs = [newton(x, fallback="steepest_descent") for x in [(-2, 1), (0, 0), (2, -2), (1, 1)]]
    for i, result in enumerate(runs):
        save_history(root / "tables" / f"ex1_part_b_newton_{i}.csv", result)
    write_csv(root / "tables/ex1_part_b_summary.csv", [summary(r, run=i) for i, r in enumerate(runs)])
    return runs


if __name__ == "__main__":
    module_cli(run)
