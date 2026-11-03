"""2(d): fixed-step residual gradient descent; the shared loop also supports 2(e)."""

import numpy as np
from .model import F, jacobian, phi_grid
from sc_cw1.common.utilities import output_dirs, plt, save_figure, save_history, module_cli


def gradient_descent(x0=(0., -1.), step_size=.05, tol=1e-6, maxit=200, gamma=0.,
                     residual=F, jac=jacobian):
    """Stop at ||F||₂<tol; gamma=0 is ordinary descent, gamma>0 is momentum."""
    if step_size <= 0 or tol <= 0 or maxit < 0 or not 0 <= gamma < 1:
        raise ValueError("invalid step size, tolerance, iteration limit or momentum")
    k = 0
    x = np.array(x0, dtype=float)
    d_prev = np.zeros(2)  # Momentum change 1: start with zero previous direction.
    iterates = [x.copy()]
    directions = [d_prev.copy()]
    alphas = [0.0]
    status = "max_iterations"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual_value = residual(x)
        g = 2*jac(x).T @ residual_value
        values = [residual_value @ residual_value]
        residual_norm = np.linalg.norm(residual_value)
        residual_norms = [residual_norm]
        gradient_norms = [np.linalg.norm(g)]
        if not np.isfinite(values[-1]) or not np.isfinite(gradient_norms[-1]):
            status = "numerical_failure"
        while residual_norm >= tol and k < maxit and status != "numerical_failure":
            # Momentum change 2: gamma=0 gives ordinary gradient descent.
            d = gamma*d_prev + g
            x_new = x-step_size*d
            # Evaluate F and its analytical residual gradient at the new point.
            residual_new = residual(x_new)
            g_new = 2*jac(x_new).T @ residual_new
            residual_norm_new = np.linalg.norm(residual_new)
            grad_norm_new = np.linalg.norm(g_new)
            value_new = residual_new @ residual_new
            if (not np.isfinite(residual_norm_new) or not np.isfinite(grad_norm_new)
                    or not np.isfinite(value_new)):
                status = "numerical_failure"
                break
            x, residual_value, g = x_new, residual_new, g_new
            residual_norm = residual_norm_new
            iterates.append(x.copy())
            values.append(value_new)
            residual_norms.append(residual_norm_new)
            gradient_norms.append(grad_norm_new)
            d_prev = d.copy()  # Momentum change 3: retain the accepted direction.
            directions.append(d.copy())
            alphas.append(step_size)
            k += 1
    if status != "numerical_failure" and residual_norm < tol:
        status = "converged"
    return dict(x=iterates[-1], iterates=np.array(iterates), values=np.array(values),
                residual_norms=np.array(residual_norms), gradient_norms=np.array(gradient_norms),
                directions=np.array(directions), alphas=np.array(alphas),
                step_kinds=["initial"]+["momentum" if gamma else "gradient_descent"]*(len(iterates)-1),
                iterations=k, status=status)


def residual_background(ax, xlim=(-.1, 1.6), ylim=(-1.15, .15)):
    """Plot residual contours behind the iteration path, as in Unit 11."""
    X, Y = np.meshgrid(np.linspace(*xlim, 350), np.linspace(*ylim, 350))
    ax.contour(X, Y, phi_grid(X, Y), levels=np.geomspace(1e-5, 10, 16),
               colors=".65", linewidths=.55)
    ax.set(xlim=xlim, ylim=ylim, xlabel="$x$", ylabel="$y$", aspect="equal")


def run(output="results"):
    """Run exactly the 2(d) settings and save the trajectory and history."""
    root = output_dirs(output)
    result = gradient_descent()
    save_history(root/"tables/ex2_part_d_gradient_descent_history.csv", result)
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    residual_background(ax)
    path = result["iterates"]
    ax.plot(path[:, 0], path[:, 1], "o-", markersize=2.3, linewidth=1, label="Gradient descent")
    ax.scatter(*path[0], marker="s", color="black", label="Start", zorder=4)
    ax.scatter(*path[-1], marker="*", color="C0", s=130, edgecolors="black", label="Final point", zorder=5)
    ax.set_title(f"2(d): step 0.05; {result['status']}, {result['iterations']} updates")
    ax.legend(fontsize=8)
    save_figure(fig, root/"figures/ex2_part_d_gradient_descent.png")
    return result


if __name__ == "__main__":
    module_cli(run)
