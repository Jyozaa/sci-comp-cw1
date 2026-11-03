"""1(b): Newton optimisation with objective Armijo step-halving."""

import numpy as np
from .model import objective, gradient, hessian
from sc_cw1.common.utilities import module_cli


def armijo(fun, x, value, g, delta, c1=1e-4, max_backtracks=50):
    """Find alpha with J(x+alpha*delta)<=J(x)+c1*alpha*g.T*delta."""
    slope = g @ delta
    if not np.isfinite(slope) or slope >= 0:
        return None  # Armijo requires a descent direction.
    alpha = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_backtracks+1):
            candidate = fun(x+alpha*delta)
            if np.isfinite(candidate) and candidate <= value+c1*alpha*slope:
                return alpha
            alpha *= 0.5  # Reduce the step until sufficient decrease holds.
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
    iterates = [x.copy()]
    alphas = [0.0]
    step_types = ["initial"]
    directions = [np.zeros(2)]
    status = "max_iterations"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value, g = fun(x), grad(x)
        grad_norm = np.linalg.norm(g)
        values, gradient_norms = [value], [grad_norm]
        if not np.all(np.isfinite(x)) or not np.isfinite(value) or not np.isfinite(grad_norm):
            status = "numerical_failure"
        for _ in range(maxit):
            if status == "numerical_failure" or grad_norm < tol:
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
            x_new = x+alpha*delta
            J_new, g_new = fun(x_new), grad(x_new)
            grad_norm_new = np.linalg.norm(g_new)
            if (not np.isfinite(J_new) or not np.isfinite(grad_norm_new)
                    or not np.all(np.isfinite(x_new)) or np.array_equal(x_new, x)):
                status = "numerical_failure"
                break
            x, value, g = x_new, J_new, g_new
            grad_norm = grad_norm_new
            iterates.append(x.copy())
            values.append(value)
            gradient_norms.append(grad_norm_new)
            alphas.append(alpha)
            step_types.append("newton")
            directions.append(delta.copy())
    if status != "numerical_failure" and grad_norm < tol:
        status = "converged"
    return dict(x=iterates[-1], iterates=np.array(iterates), values=np.array(values),
                gradient_norms=np.array(gradient_norms), alphas=np.array(alphas),
                step_kinds=step_types, directions=np.array(directions),
                iterations=len(iterates)-1, status=status)


def run(output="results"):
    """Demonstrate 1(b); the full different-start experiment is 1(c)."""
    result = newton((-2., 1.))
    print(result["status"], result["iterations"], result["x"])
    return result


if __name__ == "__main__":
    module_cli(run)
