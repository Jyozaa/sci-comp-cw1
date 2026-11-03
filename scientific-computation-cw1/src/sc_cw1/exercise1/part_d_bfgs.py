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
    B_y = B @ y
    B_new = B + ((sy + y @ B_y) / sy**2) * np.outer(s, s)
    B_new -= (np.outer(B_y, s) + np.outer(s, y @ B)) / sy
    if not np.all(np.isfinite(B_new)):
        raise ValueError("nonfinite inverse-Hessian update")
    return B_new


def bfgs(x0, omega=1.0, tol=1e-6, maxit=200, line_search=True,
         fun=objective, grad=gradient):
    """Use delta=-B*grad and stop at ||grad||₂<tol; invalid updates fail explicitly."""
    if omega <= 0 or tol <= 0 or maxit < 0:
        raise ValueError("omega/tol must be positive and maxit nonnegative")
    k = 0
    x = np.array(x0, dtype=float)
    B = omega*np.eye(2)
    iterates = [x.copy()]
    alphas = [0.0]
    directions = [np.zeros(2)]
    status = "max_iterations"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value, g = fun(x), grad(x)
        grad_norm = np.linalg.norm(g)
        values, gradient_norms = [value], [grad_norm]
        if not np.all(np.isfinite(x)) or not np.isfinite(value) or not np.isfinite(grad_norm):
            status = "numerical_failure"
        while grad_norm >= tol and k < maxit and status != "numerical_failure":
            # Calculate the direction using the inverse-Hessian approximation.
            delta = -B @ g
            alpha = armijo(fun, x, value, g, delta) if line_search else 1.0
            if alpha is None:
                status = "line_search_failure"
                break
            # Evaluate the trial step and its new gradient.
            s = alpha*delta
            x_new = x+s
            J_new, g_new = fun(x_new), grad(x_new)
            grad_norm_new = np.linalg.norm(g_new)
            if (not np.isfinite(J_new) or not np.isfinite(grad_norm_new)
                    or not np.all(np.isfinite(x_new)) or np.array_equal(x_new, x)):
                status = "numerical_failure"
                break
            # The gradient difference supplies the curvature information.
            y = g_new-g
            x, value, g = x_new, J_new, g_new
            grad_norm = grad_norm_new
            iterates.append(x.copy())
            values.append(value)
            gradient_norms.append(grad_norm_new)
            alphas.append(alpha)
            directions.append(delta.copy())
            k += 1
            if grad_norm_new < tol:
                break  # No further direction needs B after convergence.
            try:
                # Apply the prescribed inverse-BFGS update.
                B = inverse_bfgs_update(B, s, y)
            except ValueError:
                status = "numerical_failure"
                break
    if status != "numerical_failure" and grad_norm < tol:
        status = "converged"
    return dict(x=iterates[-1], iterates=np.array(iterates), values=np.array(values),
                gradient_norms=np.array(gradient_norms), alphas=np.array(alphas),
                directions=np.array(directions), step_kinds=["initial"]+["bfgs"]*(len(iterates)-1),
                iterations=k, status=status)


def run(output="results"):
    """Demonstrate 1(d); 1(e) supplies the required convergence comparison."""
    result = bfgs((-2., 1.), omega=.01)
    print(result["status"], result["iterations"], result["x"])
    return result


if __name__ == "__main__":
    module_cli(run)
