"""1(c): multiple starts, distinct stationary points and objective contours."""

import numpy as np
from .model import objective, gradient, hessian, objective_grid
from .part_b_newton import newton
from sc_cw1.common.utilities import output_dirs, plt, save_figure, write_csv, module_cli


def classify_stationary_point(x, tol=1e-6):
    """Use the two-variable second-derivative test, after checking stationarity."""
    gnorm = np.linalg.norm(gradient(x))
    H = hessian(x)
    determinant = np.linalg.det(H)
    label = "inconclusive"
    if gnorm >= tol:
        label = "not_stationary"
    elif determinant < -1e-7:
        label = "saddle"
    elif determinant > 1e-7 and H[0, 0] > 1e-7:
        label = "minimum"
    elif determinant > 1e-7 and H[0, 0] < -1e-7:
        label = "maximum"
    return dict(classification=label, gradient_norm=gnorm,
                hessian_determinant=determinant, hessian_11=H[0, 0])


def distinct_stationary_points(runs, distance_tol=1e-4):
    """Merge converged points separated by less than distance_tol in x."""
    good = [r for r in runs if r["status"] == "converged"]
    good.sort(key=lambda r: r["gradient_norms"][-1])
    points = []
    for result in good:
        x = result["x"]
        info = classify_stationary_point(x)
        if info["classification"] == "not_stationary":
            continue
        for point in points:
            if np.linalg.norm(x-np.array([point["x1"], point["x2"]])) < distance_tol:
                point["runs_in_cluster"] += 1
                break
        else:
            points.append(dict(x1=x[0], x2=x[1], value=objective(x),
                               **info, runs_in_cluster=1))
    points.sort(key=lambda p: p["value"])
    minima = [p for p in points if p["classification"] == "minimum"]
    for i, point in enumerate(points):
        point["point_id"] = i+1
        point["lowest_minimum_found"] = bool(minima and point is minima[0])
    return points


def multistart(grid_size=21):
    """Run Newton/Armijo from a grid spanning the required square."""
    axis = np.linspace(-2, 2, grid_size)
    starts = [(u, v) for u in axis for v in axis]
    # The optional -gradient safeguard is taught gradient descent, recorded separately.
    runs = [newton(x, fallback=True) for x in starts]
    return starts, runs, distinct_stationary_points(runs)


def run(output="results", grid_size=21):
    """Save the different-start results, stationary points and required contours."""
    root = output_dirs(output)
    starts, runs, points = multistart(grid_size)
    rows = []
    for x0, r in zip(starts, runs):
        rows.append(dict(start_x1=x0[0], start_x2=x0[1], x1=r["x"][0], x2=r["x"][1],
                         value=r["values"][-1], gradient_norm=r["gradient_norms"][-1],
                         iterations=r["iterations"], status=r["status"],
                         newton_steps=r["step_kinds"].count("newton"),
                         fallback_steps=sum(k.startswith("fallback") for k in r["step_kinds"]),
                         classification=classify_stationary_point(r["x"])["classification"]
                         if r["status"] == "converged" else "not_classified"))
    write_csv(root/"tables/ex1_part_c_multistart.csv", rows)
    write_csv(root/"tables/ex1_part_c_stationary_points.csv", points)
    X, Y = np.meshgrid(np.linspace(-2, 2, 501), np.linspace(-2, 2, 501))
    Z = objective_grid(X, Y)
    levels = np.r_[np.linspace(Z.min()+.05, 2, 12), [3, 5, 8, 12, 20, 35, 55]]
    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=.8)
    fig.colorbar(cs, ax=ax, label="Objective J")
    for p in points:
        best = p["lowest_minimum_found"]
        marker = "*" if best else ("o" if p["classification"] == "minimum" else "x")
        label = "Lowest minimum found (global candidate)" if best else p["classification"]
        ax.scatter(p["x1"], p["x2"], marker=marker, s=140 if best else 55,
                   color="crimson" if best else "darkorange", label=label, zorder=4)
        ax.annotate(str(p["point_id"]), (p["x1"], p["x2"]), xytext=(7, 7), textcoords="offset points")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title="1(c): objective contours and stationary points",
           xlim=(-2, 2), ylim=(-2, 2), aspect="equal")
    ax.legend(fontsize=8)
    save_figure(fig, root/"figures/ex1_part_c_contours.png")
    return dict(starts=starts, runs=runs, points=points)


if __name__ == "__main__":
    module_cli(run)
