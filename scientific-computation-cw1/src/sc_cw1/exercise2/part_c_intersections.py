"""2(c): actual intersections of the plotted piecewise linear zero curves."""

import numpy as np
from sc_cw1.common.utilities import output_dirs, pyplot, save_figure, write_csv, module_cli
from .model import F


def cross2(a, b):
    """2D scalar cross product, broadcasting over arrays of segment vectors."""
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def polyline_intersections(curve1, curve2, distance_tol=1e-5):
    """Find segment intersections p+t*r=q+u*s, with 0<=t,u<=1.

    This operates on the plotted curves, so the resulting coordinates are
    graphical discretisation estimates, not results of a root solver. Parallel
    and coincident segments are excluded; these zero curves have transverse
    crossings. Duplicate adjacent-segment hits are merged in Euclidean distance.
    """
    q = curve2[:-1]
    s = np.diff(curve2, axis=0)
    hits = []
    for p, r in zip(curve1[:-1], np.diff(curve1, axis=0)):
        den = cross2(r, s)
        mask = np.abs(den) > 1e-14
        qp = q[mask] - p
        t = cross2(qp, s[mask]) / den[mask]
        u = cross2(qp, r) / den[mask]
        for j in np.flatnonzero((t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
            hit = p + t[j]*r
            if all(np.linalg.norm(hit - previous) > distance_tol for previous in hits):
                hits.append(hit)
    return sorted(hits, key=lambda x: x[0])


def graphical_estimates(samples=2001):
    """Sample the circle (x-1)²+y²=1 and y=cos(2x); intersect the polylines."""
    if not isinstance(samples, int) or samples < 100:
        raise ValueError("use at least 100 curve samples")
    theta = np.linspace(0, 2*np.pi, samples)
    circle = np.column_stack((1+np.cos(theta), np.sin(theta)))
    x = np.linspace(-2, 2, samples)
    cosine = np.column_stack((x, np.cos(2*x)))
    return circle, cosine, polyline_intersections(circle, cosine)


def refine_intersections(estimates, tol=1e-12, maxit=100):
    """Supplementary scalar bisection near graphical estimates, never their replacement.

    Eliminating y gives q(x)=(x-1)²+cos²(2x)-1=0 on [0,2]. A bracket of
    +/-0.02 is attempted around each estimate. Failed brackets are reported.
    No completeness claim about all roots follows from sampling/sign changes.
    """
    def q(x):
        return (x-1)**2 + np.cos(2*x)**2 - 1

    rows = []
    for i, estimate in enumerate(estimates, 1):
        a, b = max(0., estimate[0]-.02), min(2., estimate[0]+.02)
        fa, fb = q(a), q(b)
        status = "unbracketed"
        x = float(estimate[0])
        k = 0
        if fa*fb <= 0:
            status = "max_iterations"
            for k in range(1, maxit+1):
                x = (a+b)/2
                fx = q(x)
                if abs(fx) < tol:
                    status = "converged"
                    break
                if fa*fx <= 0:
                    b = x
                else:
                    a, fa = x, fx
        point = np.array([x, np.cos(2*x)])
        f = F(point)
        rows.append(dict(intersection=i, x1=float(point[0]), x2=float(point[1]),
                         f1=float(f[0]), f2=float(f[1]), residual_norm=float(np.linalg.norm(f)),
                         status=status, iterations=k, source="supplementary scalar bisection"))
    return rows


def run(output="results"):
    """Save zero-curve figure, graphical intersection table and separate refinement."""
    root = output_dirs(output)
    circle, cosine, estimates = graphical_estimates()
    rows = []
    for i, x in enumerate(estimates, 1):
        f = F(x)
        rows.append(dict(intersection=i, x1=float(x[0]), x2=float(x[1]), f1=float(f[0]),
                         f2=float(f[1]), residual_norm=float(np.linalg.norm(f)),
                         source="intersection of plotted 2001-sample polylines"))
    refined = refine_intersections(estimates)
    write_csv(root / "tables/ex2_part_c_graphical_intersections.csv", rows)
    write_csv(root / "tables/ex2_part_c_refined_intersections.csv", refined)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    ax.plot(*circle.T, color="C0", label=r"$F_1=0$: $(x-1)^2+y^2=1$")
    ax.plot(*cosine.T, color="C1", label=r"$F_2=0$: $y=\cos(2x)$")
    for i, x in enumerate(estimates, 1):
        ax.scatter(*x, c="crimson", s=45, zorder=4)
        ax.annotate(f"{i}: ({x[0]:.3f}, {x[1]:.3f})", x,
                    xytext=(-8 if x[0] > 1 else 8, 10),
                    ha="right" if x[0] > 1 else "left", textcoords="offset points", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1.5))
    ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="$x$", ylabel="$y$", aspect="equal",
           title=f"2(c): zero curves; {len(estimates)} graphical intersections")
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    save_figure(fig, root / "figures/ex2_part_c_intersections.png")
    return dict(graphical=rows, refined=refined)


if __name__ == "__main__":
    module_cli(run)
