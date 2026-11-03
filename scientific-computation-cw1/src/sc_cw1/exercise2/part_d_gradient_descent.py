"""2(d): direct fixed-step descent with explicit residual-norm stopping."""

import numpy as np
from sc_cw1.common.utilities import (
    NUMERICAL_ERRORS, SolverResult, finite, vector, validate_controls,
    output_dirs, pyplot, plot_path, save_figure, save_history, summary, write_csv, module_cli,
)
from .model import F, jacobian, phi_grid


def residual_state(x, residual, jac):
    """Compute F, φ=F.T F, ∇φ=2J.T F and their Euclidean norms."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        finite(x)
        f = vector(finite(residual(x)))
        J = np.asarray(finite(jac(x)), dtype=float)
        if J.shape != (2, 2):
            raise ValueError("Jacobian must have shape (2,2)")
        g = finite(2*J.T @ f)
        value = float(finite(f @ f))
        rn, gn = float(finite(np.linalg.norm(f))), float(finite(np.linalg.norm(g)))
    return f, value, g, rn, gn


def fixed_descent(x0=(0., -1.), *, step_size=.05, tol=1e-6, maxit=200,
                  residual=F, jac=jacobian, gamma=0., initial_direction=None):
    """Shared loop for 2(d/e): d_k=gamma*d_{k-1}+2J.T F; x+=-lambda*d_k.

    The gamma=0 case is ordinary gradient descent. Momentum supplies a gamma
    strictly between zero and one via the 2(e) wrapper. d_{-1}=0 by convention.
    Stop when ||F||_2<tol (Unit 10, slide 10; not specified explicitly by brief).
    Exactly maxit updates at most, with convergence checked after the final one.
    No line search, clipping, velocity reset or early gradient-norm stopping.
    """
    validate_controls(tol, maxit)
    x = vector(x0)
    if not np.isfinite(step_size) or step_size <= 0 or not 0 <= gamma < 1:
        raise ValueError("step_size>0 must be finite; gamma must be in [0,1)")
    # MOMENTUM CHANGE 1: remember the previous direction; zero is an assumption.
    previous = np.zeros(2) if initial_direction is None else vector(initial_direction)
    result = SolverResult(parameters=dict(step_size=step_size, gamma=gamma, tol=tol,
        maxit=maxit, initial_direction1=float(previous[0]), initial_direction2=float(previous[1]),
        stopping_statistic="residual_norm"))
    try:
        finite(previous)
        f, value, g, rn, gn = residual_state(x, residual, jac)
    except NUMERICAL_ERRORS as exc:
        return result.stop("numerical_failure", f"initial evaluation: {exc}")
    result.append(x, value, gn, residual_norm=rn, f1=float(f[0]), f2=float(f[1]),
                  direction1=float(previous[0]), direction2=float(previous[1]), step_kind="initial")
    for _ in range(maxit):
        if rn < tol:
            return result.stop("converged", "residual norm < tolerance")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                # MOMENTUM CHANGE 2: accumulate previous gradient information.
                # gamma=0 gives d_k=grad_phi, the original descent direction.
                direction = finite(gamma*previous + g)
                xn = finite(x - step_size*direction)
            fn, vn, gnew, rnnew, gnnew = residual_state(xn, residual, jac)
        except NUMERICAL_ERRORS as exc:
            return result.stop("numerical_failure", str(exc))
        result.append(xn, vn, gnnew, step_size, residual_norm=rnnew,
                      f1=float(fn[0]), f2=float(fn[1]), direction1=float(direction[0]),
                      direction2=float(direction[1]), step_kind="momentum" if gamma else "gradient_descent")
        # MOMENTUM CHANGE 3: retain d_k for the next iteration (no reset).
        previous = direction.copy()
        x, f, value, g, rn, gn = xn, fn, vn, gnew, rnnew, gnnew
    if rn < tol:
        return result.stop("converged", "residual norm < tolerance at final allowed update")
    return result


def gradient_descent(x0=(0., -1.), *, step_size=.05, tol=1e-6, maxit=200,
                     residual=F, jac=jacobian):
    """Ordinary descent: x_{k+1}=x_k-step_size*(2J.T F), no momentum."""
    return fixed_descent(x0, step_size=step_size, tol=tol, maxit=maxit,
                         residual=residual, jac=jac, gamma=0.)


def add_residual_background(ax, xlim=(-.1, 1.6), ylim=(-1.15, .15)):
    """Add φ contours as context for trajectories, including narrow valleys."""
    X, Y = np.meshgrid(np.linspace(*xlim, 350), np.linspace(*ylim, 350))
    ax.contour(X, Y, phi_grid(X, Y), levels=np.geomspace(1e-5, 10, 16),
               colors="0.65", linewidths=.55, alpha=.7)
    ax.set(xlim=xlim, ylim=ylim, xlabel="$x$", ylabel="$y$", aspect="equal")


def run(output="results"):
    """Run 2(d) with exactly lambda=.05, x0=(0,-1), tol=1e-6, maxit=200."""
    root = output_dirs(output)
    result = gradient_descent()
    save_history(root / "tables/ex2_part_d_gradient_descent_history.csv", result)
    write_csv(root / "tables/ex2_part_d_summary.csv", [summary(result)])
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    add_residual_background(ax)
    plot_path(ax, result, "Gradient descent")
    ax.set_title(f"2(d): fixed step 0.05; {result.status}, {result.nit} steps\n"
                 f"Final residual norm = {result.history[-1]['residual_norm']:.3e}")
    ax.legend(fontsize=8)
    save_figure(fig, root / "figures/ex2_part_d_gradient_descent.png")
    return result


if __name__ == "__main__":
    module_cli(run)
