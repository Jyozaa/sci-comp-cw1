"""1(b): Newton optimisation with objective Armijo step-halving."""

import numpy as np
from .model import objective, gradient, hessian
from sc_cw1.common.utilities import module_cli


def armijo(fun, x, value, g, delta, c1=1e-4, max_backtracks=50):
    """Find alpha with J(x+alpha*delta)<=J(x)+c1*alpha*g.T*delta."""
    slope = g @ delta
    if not np.isfinite(slope) or slope >= 0:
        return None
    alpha = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_backtracks+1):
            candidate = fun(x+alpha*delta)
            if np.isfinite(candidate) and candidate <= value+c1*alpha*slope:
                return alpha
            alpha *= 0.5
    return None  # Do not accept a step after a failed search.


def newton(x0, tol=1e-6, maxit=200, line_search=True,
           fun=objective, grad=gradient, hess=hessian):
    """Solve H*delta=-grad; stop at ||grad||₂<tol or after maxit updates.

    An unusable Newton direction fails the run; no other direction is used.
    Fixed comparison runs use line_search=False and exactly unit steps.
    """
    if tol <= 0 or maxit < 0:
        raise ValueError("tol must be positive and maxit nonnegative")
    x = np.array(x0, dtype=float)
    xs, alphas, kinds, directions = [x.copy()], [0.0], ["initial"], [np.zeros(2)]
    status = "max_iterations"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value, g = fun(x), grad(x)
        values, norms = [value], [np.linalg.norm(g)]
        if not np.all(np.isfinite(x)) or not np.isfinite(value) or not np.isfinite(norms[-1]):
            status = "numerical_failure"
        for _ in range(maxit):
            if status == "numerical_failure" or norms[-1] < tol:
                break
            H = hess(x)
            if not np.all(np.isfinite(H)):
                status = "numerical_failure"
                break
            try:
                if np.linalg.cond(H) > 1e12:
                    raise np.linalg.LinAlgError("singular or poorly conditioned Hessian")
                # Compute the Newton direction by solving H*delta=-grad.
                delta = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                status = "numerical_failure"
                break
            if line_search and g @ delta >= 0:
                status = "line_search_failure"
                break
            alpha = armijo(fun, x, value, g, delta) if line_search else 1.0
            if alpha is None:
                status = "line_search_failure"
                break
            xn = x+alpha*delta
            vn, gn = fun(xn), grad(xn)
            nn = np.linalg.norm(gn)
            if (not np.isfinite(vn) or not np.isfinite(nn)
                    or not np.all(np.isfinite(xn)) or np.array_equal(xn, x)):
                status = "numerical_failure"
                break
            x, value, g = xn, vn, gn
            xs.append(x.copy())
            values.append(value)
            norms.append(nn)
            alphas.append(alpha)
            kinds.append("newton")
            directions.append(delta.copy())
    if status != "numerical_failure" and norms[-1] < tol:
        status = "converged"
    return dict(x=xs[-1], iterates=np.array(xs), values=np.array(values),
                gradient_norms=np.array(norms), alphas=np.array(alphas),
                step_kinds=kinds, directions=np.array(directions),
                iterations=len(xs)-1, status=status)


def run(output="results"):
    """Demonstrate 1(b); the full different-start experiment is 1(c)."""
    result = newton((-2., 1.))
    print(result["status"], result["iterations"], result["x"])
    return result


if __name__ == "__main__":
    module_cli(run)
