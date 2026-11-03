"""2(b): contour-visible local-minimum estimates with supplementary refinement."""

import numpy as np
from sc_cw1.common.utilities import output_dirs, pyplot, save_figure, summary, write_csv, module_cli
from sc_cw1.exercise1.part_b_newton import newton
from sc_cw1.exercise1.part_c_minima import distinct_stationary_points
from .model import F, phi, grad_phi, hessian_phi, phi_grid


def grid_minimum_candidates(grid_size=401):
    """Return eight-neighbour interior grid minima of φ on [-2,2]^2.

    These are finite-grid estimates, not an exhaustive theorem. Boundary pixels
    are excluded because constrained boundary minima are not stationary points
    of the unconstrained residual objective. Full region is still plotted.
    """
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 5:
        raise ValueError("grid_size must be an integer >=5")
    axis = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(axis, axis)
    Z = phi_grid(X, Y)
    centre = Z[1:-1, 1:-1]
    mask = np.ones(centre.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                mask &= centre < Z[1+di:grid_size-1+di, 1+dj:grid_size-1+dj]
    ii, jj = np.where(mask)
    return np.column_stack((axis[jj+1], axis[ii+1]))


def estimate_minima(grid_size=401):
    """Refine contour/grid candidates, classifying with exact ∇²φ.

    Supplementary Newton/Armijo applies to ∇φ=0, not F=0. It therefore admits
    positive-residual local minima and explicitly checks whether F is small.
    """
    candidates = grid_minimum_candidates(grid_size)
    runs = [newton(x, fun=phi, grad=grad_phi, hess=hessian_phi,
                   tol=1e-7, maxit=200, fallback="steepest_descent") for x in candidates]
    points = distinct_stationary_points(runs, fun=phi, grad=grad_phi, hess=hessian_phi,
                                        gradient_tol=1e-7, distance_tol=1e-4)
    points = [p for p in points if p["classification"] == "minimum"
              and -2 <= p["x1"] <= 2 and -2 <= p["x2"] <= 2]
    for p in points:
        f = F((p["x1"], p["x2"]))
        p.update(f1=float(f[0]), f2=float(f[1]), residual_norm=float(np.linalg.norm(f)),
                 is_root=bool(np.linalg.norm(f) < 1e-6), source="grid estimate; supplementary Newton refinement")
        # The 'lowest candidate' flag from the generic clustering is inapplicable
        # to the task's count of apparent minima, so retain only the diagnostics.
        p.pop("lowest_minimum_found")
    return candidates, runs, points


def run(output="results", grid_size=401):
    """Plot φ contours and save apparent minima, raw estimates and refinements."""
    root = output_dirs(output)
    candidates, runs, points = estimate_minima(grid_size)
    write_csv(root / "tables/ex2_part_b_grid_candidates.csv",
              [summary(r, grid_x1=float(x[0]), grid_x2=float(x[1]), grid_size=grid_size,
                       grid_phi=phi(x)) for x, r in zip(candidates, runs)])
    write_csv(root / "tables/ex2_part_b_minima.csv", points)
    plt = pyplot()
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterSciNotation
    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    X, Y = np.meshgrid(np.linspace(-2, 2, grid_size), np.linspace(-2, 2, grid_size))
    levels = np.geomspace(1e-5, 120, 23)
    cs = ax.contour(X, Y, phi_grid(X, Y), levels=levels, cmap="viridis",
                    norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
    fig.colorbar(cs, ax=ax, ticks=10.**np.arange(-5, 3), format=LogFormatterSciNotation(),
                 label=r"$\phi=F_1^2+F_2^2$ (geometric contour levels)")
    for i, p in enumerate(points, 1):
        ax.scatter(p["x1"], p["x2"], color="crimson" if p["is_root"] else "darkorange",
                   marker="o" if p["is_root"] else "D", s=60, zorder=5,
                   label=f"Estimate {i}: {'root' if p['is_root'] else 'positive residual minimum'}")
        ax.annotate(str(i), (p["x1"], p["x2"]), xytext=(7, 7), textcoords="offset points")
    ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="$x$", ylabel="$y$", aspect="equal",
           title=f"2(b): squared residual contours; {len(points)} apparent local minima")
    ax.legend(fontsize=8, loc="upper left")
    save_figure(fig, root / "figures/ex2_part_b_residual_contours.png")
    return dict(candidates=candidates, runs=runs, points=points)


if __name__ == "__main__":
    module_cli(run)
