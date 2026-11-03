"""2(d): fixed-step residual gradient descent; the shared loop also supports 2(e)."""

import numpy as np
from .model import F, jacobian, phi_grid
from sc_cw1.common.utilities import output_dirs, plt, save_figure, save_history, module_cli


def gradient_descent(x0=(0., -1.), step_size=.05, tol=1e-6, maxit=200, gamma=0.,
                     residual=F, jac=jacobian):
    """Stop at ||F||₂<tol; gamma=0 is ordinary descent, gamma>0 is momentum."""
    if step_size <= 0 or tol <= 0 or maxit < 0 or not 0 <= gamma < 1:
        raise ValueError("invalid step size, tolerance, iteration limit or momentum")
    x = np.array(x0, dtype=float)
    previous = np.zeros(2)  # Momentum change 1: choose the initial previous direction to be zero.
    xs, directions, alphas = [x.copy()], [previous.copy()], [0.0]
    status = "max_iterations"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        f = residual(x)
        g = 2*jac(x).T @ f
        values, residual_norms, gradient_norms = [f @ f], [np.linalg.norm(f)], [np.linalg.norm(g)]
        if not np.isfinite(values[-1]) or not np.isfinite(gradient_norms[-1]):
            status = "numerical_failure"
        for _ in range(maxit):
            if status == "numerical_failure" or residual_norms[-1] < tol:
                break
            # Momentum change 2: gamma=0 gives the ordinary direction d=g.
            d = gamma*previous + g
            xn = x-step_size*d
            fn = residual(xn)
            gn = 2*jac(xn).T @ fn
            rn, nn, vn = np.linalg.norm(fn), np.linalg.norm(gn), fn @ fn
            if not np.isfinite(rn) or not np.isfinite(nn) or not np.isfinite(vn):
                status = "numerical_failure"
                break
            x, f, g = xn, fn, gn
            xs.append(x.copy())
            values.append(vn)
            residual_norms.append(rn)
            gradient_norms.append(nn)
            previous = d.copy()  # Momentum change 3: remember d for the next update.
            directions.append(d.copy())
            alphas.append(step_size)
    if status != "numerical_failure" and residual_norms[-1] < tol:
        status = "converged"
    return dict(x=xs[-1], iterates=np.array(xs), values=np.array(values),
                residual_norms=np.array(residual_norms), gradient_norms=np.array(gradient_norms),
                directions=np.array(directions), alphas=np.array(alphas),
                step_kinds=["initial"]+["momentum" if gamma else "gradient_descent"]*(len(xs)-1),
                iterations=len(xs)-1, status=status)


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
