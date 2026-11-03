"""1(c): deterministic multistart, stationary-point classification and contours."""

import numpy as np
from sc_cw1.common.utilities import (
    checked_evaluation, finite, finite_difference_jacobian, output_dirs, pyplot,
    save_figure, summary, write_csv, module_cli,
)
from .model import objective, gradient, hessian, objective_grid
from .part_b_newton import newton
from .global_analysis import run as run_global_analysis


def classify_stationary_point(x, *, grad=gradient, hess=hessian,
                              gradient_tol=1e-6, eigenvalue_tol=1e-7):
    """Classify an unconstrained point from ||g|| and symmetric Hessian eigenvalues.

    Positive eigenvalues certify a strict local minimum at an exact stationary
    point. Near-zero eigenvalues are inconclusive; a small gradient alone is
    insufficient. Tolerances are absolute and the point remains numerical.
    """
    gn = float(finite(np.linalg.norm(grad(x))))
    H = finite(hess(x))
    eig = np.linalg.eigvalsh((H + H.T) / 2)
    if gn >= gradient_tol:
        label = "not_stationary"
    elif eig[0] > eigenvalue_tol:
        label = "minimum"
    elif eig[-1] < -eigenvalue_tol:
        label = "maximum"
    elif eig[0] < -eigenvalue_tol and eig[-1] > eigenvalue_tol:
        label = "saddle"
    else:
        label = "inconclusive"
    return dict(classification=label, gradient_norm=gn,
                eigenvalue_min=float(eig[0]), eigenvalue_max=float(eig[-1]))


def distinct_stationary_points(runs, *, fun=objective, grad=gradient, hess=hessian,
                               distance_tol=1e-4, gradient_tol=1e-6):
    """Cluster only converged, stationary runs within distance_tol in Euclidean x.

    Representatives are sorted by increasing gradient norm before clustering,
    so the most accurate point is retained. 1e-4 is much smaller than observed
    separation, while allowing small gradient-to-position amplification.
    """
    if not np.isfinite(distance_tol) or distance_tol <= 0:
        raise ValueError("distance_tol must be positive and finite")
    accepted = sorted((r for r in runs if r.converged),
                      key=lambda r: np.linalg.norm(grad(r.x)))
    clusters = []
    for result in accepted:
        x = result.x
        info = classify_stationary_point(x, grad=grad, hess=hess, gradient_tol=gradient_tol)
        if info["classification"] == "not_stationary":
            continue
        duplicate = next((c for c in clusters if np.linalg.norm(
            x - np.array([c["x1"], c["x2"]])) < distance_tol), None)
        if duplicate is not None:
            duplicate["runs_in_cluster"] += 1
        else:
            clusters.append(dict(x1=float(x[0]), x2=float(x[1]), value=float(fun(x)),
                                 **info, status="converged", runs_in_cluster=1))
    clusters.sort(key=lambda c: c["value"])
    minima = [c for c in clusters if c["classification"] == "minimum"]
    for i, row in enumerate(clusters):
        row["point_id"] = i + 1
        row["lowest_minimum_found"] = bool(minima and row is minima[0])
    return clusters


def multistart(grid_size=21, *, tol=1e-6, maxit=200, fallback="steepest_descent"):
    """Run Newton/Armijo on a grid including all edges of [-2,2]^2.

    Explicit fallback makes this a safeguarded Newton search. It can converge
    to any stationary point and is not an exhaustive/global certification.
    """
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer >=2")
    starts = [(u, v) for u in np.linspace(-2, 2, grid_size)
              for v in np.linspace(-2, 2, grid_size)]
    runs = [newton(x, tol=tol, maxit=maxit, fallback=fallback) for x in starts]
    points = distinct_stationary_points(runs, gradient_tol=tol)
    return starts, runs, points


def run(output="results", grid_size=21):
    """Run 1(c), save every start/status, distinct points and contour figure."""
    root = output_dirs(output)
    starts, runs, points = multistart(grid_size)
    rows = []
    for x0, r in zip(starts, runs):
        row = summary(r, start_x1=x0[0], start_x2=x0[1])
        row["fallback_steps"] = sum(s.get("step_kind", "").startswith("fallback") for s in r.history)
        row["newton_steps"] = sum(s.get("step_kind") == "newton" for s in r.history)
        row["classification"] = (classify_stationary_point(r.x)["classification"]
                                 if r.converged else "not_classified_failed_run")
        rows.append(row)
    write_csv(root / "tables/ex1_part_c_multistart.csv", rows)
    write_csv(root / "tables/ex1_part_c_stationary_points.csv", points)
    # Supplementary proof leaves all multistart iterates/results unchanged.
    proof = run_global_analysis(root)
    best = next(row for row in points if row["lowest_minimum_found"])
    a = proof["global_root"]["a"]
    if np.linalg.norm(np.array([best["x1"], best["x2"]])-np.array([a, -a])) > 1e-4:
        raise RuntimeError("numerical lowest minimum does not match the proven global root")
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    X, Y = np.meshgrid(np.linspace(-2, 2, 501), np.linspace(-2, 2, 501))
    Z = objective_grid(X, Y)
    levels = np.r_[np.linspace(Z.min()+0.05, 2, 12), [3, 5, 8, 12, 20, 35, 55]]
    cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=.8)
    fig.colorbar(cs, ax=ax, label="Objective J")
    for row in points:
        best = row["lowest_minimum_found"]
        marker = "*" if best else ("o" if row["classification"] == "minimum" else "x")
        ax.scatter(row["x1"], row["x2"], marker=marker, s=160 if best else 55,
                   color="crimson" if best else "darkorange", zorder=4,
                   label="Global minimum (see proof)" if best else f"{row['classification']} {row['point_id']}")
        ax.annotate(str(row["point_id"]), (row["x1"], row["x2"]), xytext=(7, 7), textcoords="offset points")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=f"1(c): Newton multistart ({grid_size} × {grid_size} starts)",
           xlim=(-2, 2), ylim=(-2, 2), aspect="equal")
    ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, root / "figures/ex1_part_c_contours.png")
    return dict(starts=starts, runs=runs, points=points, global_analysis=proof)


if __name__ == "__main__":
    module_cli(run)
