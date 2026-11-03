"""2(b): squared-residual contours and apparent grid minima."""

import numpy as np
from matplotlib.colors import LogNorm
from .model import F, phi, phi_grid
from sc_cw1.common.utilities import output_dirs, plt, save_figure, write_csv, module_cli


def grid_minimum_candidates(grid_size=401):
    """Find grid values lower than their eight neighbouring grid values."""
    axis = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(axis, axis)
    Z = phi_grid(X, Y)
    points = []
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            neighbours = Z[i-1:i+2, j-1:j+2].copy()
            neighbours[1, 1] = np.inf  # Compare against neighbours, not itself.
            if Z[i, j] < np.min(neighbours):
                points.append([axis[j], axis[i]])
    return np.array(points).reshape(-1, 2)


def run(output="results", grid_size=401):
    """Plot phi over [-2,2]²; label apparent minima without numerical refinement."""
    root = output_dirs(output)
    points = grid_minimum_candidates(grid_size)
    rows = [dict(x=p[0], y=p[1], phi=phi(p), residual_norm=np.linalg.norm(F(p)),
                 interpretation="apparent grid minimum") for p in points]
    write_csv(root/"tables/ex2_part_b_minima.csv", rows)
    X, Y = np.meshgrid(np.linspace(-2, 2, grid_size), np.linspace(-2, 2, grid_size))
    levels = [1e-3, .01, .1, .5, 1, 2, 5, 10, 20, 50, 100]
    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    cs = ax.contour(X, Y, phi_grid(X, Y), levels=levels, cmap="viridis", norm=LogNorm())
    fig.colorbar(cs, ax=ax, format="%.1g", label=r"$\phi=F_1^2+F_2^2$")
    for i, p in enumerate(points, 1):
        ax.scatter(*p, color="crimson", s=40, zorder=4)
        ax.annotate(f"{i}: ({p[0]:.2f}, {p[1]:.2f})", p, xytext=(-7 if p[0] > 1 else 7, 7),
                    ha="right" if p[0] > 1 else "left", textcoords="offset points", fontsize=8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="$x$", ylabel="$y$", aspect="equal",
           title=f"2(b): squared-residual contours; {len(points)} apparent minima")
    save_figure(fig, root/"figures/ex2_part_b_residual_contours.png")
    return rows


if __name__ == "__main__":
    module_cli(run)
