"""1(d): the printed inverse-Hessian BFGS update, without Hessians or solves."""

import numpy as np
from .model import objective, gradient
from .part_b_newton import armijo
from sc_cw1.common.utilities import module_cli


def inverse_bfgs_update(B, s, y):
    """Update the inverse approximation using the expanded formula in the brief."""
    sy = s @ y
    if not np.isfinite(sy) or sy <= 0 or sy <= 1e-12*np.linalg.norm(s)*np.linalg.norm(y):
        raise ValueError("nonpositive or unsafe BFGS curvature")
    By = B @ y
    updated = B + ((sy + y @ By) / sy**2) * np.outer(s, s)
    updated -= (np.outer(By, s) + np.outer(s, y @ B)) / sy
    if not np.all(np.isfinite(updated)):
        raise ValueError("nonfinite inverse-Hessian update")
    return updated


def bfgs(x0, omega=1.0, tol=1e-6, maxit=200, line_search=True,
         fun=objective, grad=gradient):
    """Use delta=-B*grad and stop at ||grad||₂<tol; invalid updates fail explicitly."""
    if omega <= 0 or tol <= 0 or maxit < 0:
        raise ValueError("omega/tol must be positive and maxit nonnegative")
    x = np.array(x0, dtype=float)
    B = omega*np.eye(2)
    xs, alphas, directions = [x.copy()], [0.0], [np.zeros(2)]
    status = "max_iterations"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value, g = fun(x), grad(x)
        values, norms = [value], [np.linalg.norm(g)]
        if not np.all(np.isfinite(x)) or not np.isfinite(value) or not np.isfinite(norms[-1]):
            status = "numerical_failure"
        for _ in range(maxit):
            if status == "numerical_failure" or norms[-1] < tol:
                break
            delta = -B @ g
            alpha = armijo(fun, x, value, g, delta) if line_search else 1.0
            if alpha is None:
                status = "line_search_failure"
                break
            s = alpha*delta
            xn = x+s
            vn, gn = fun(xn), grad(xn)
            nn = np.linalg.norm(gn)
            if (not np.isfinite(vn) or not np.isfinite(nn)
                    or not np.all(np.isfinite(xn)) or np.array_equal(xn, x)):
                status = "numerical_failure"
                break
            y = gn-g
            x, value, g = xn, vn, gn
            xs.append(x.copy())
            values.append(value)
            norms.append(nn)
            alphas.append(alpha)
            directions.append(delta.copy())
            if nn < tol:
                break  # No further direction needs B after convergence.
            try:
                B = inverse_bfgs_update(B, s, y)
            except ValueError:
                status = "numerical_failure"
                break
    if status != "numerical_failure" and norms[-1] < tol:
        status = "converged"
    return dict(x=xs[-1], iterates=np.array(xs), values=np.array(values),
                gradient_norms=np.array(norms), alphas=np.array(alphas),
                directions=np.array(directions), step_kinds=["initial"]+["bfgs"]*(len(xs)-1),
                iterations=len(xs)-1, status=status)


def run(output="results"):
    """Demonstrate 1(d); 1(e) supplies the required convergence comparison."""
    result = bfgs((-2., 1.), omega=.01)
    print(result["status"], result["iterations"], result["x"])
    return result


if __name__ == "__main__":
    module_cli(run)
