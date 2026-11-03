"""2(c): draw the two zero curves and estimate their graphical crossings."""

import numpy as np
from .model import F
from sc_cw1.common.utilities import output_dirs, plt, save_figure, write_csv, module_cli


def graphical_estimates(samples=2001):
    """Estimate crossings by one linear interpolation between plotted samples."""
    theta = np.linspace(0, 2*np.pi, samples)
    circle = np.column_stack((1+np.cos(theta), np.sin(theta)))
    x = np.linspace(-2, 2, samples)
    y = np.cos(2*x)
    cosine = np.column_stack((x, y))
    # Along the cosine curve, a sign change of the circle residual brackets a crossing.
    q = (x-1)**2+y*y-1
    estimates = []
    for i in range(samples-1):
        if q[i]*q[i+1] < 0:
            t = q[i]/(q[i]-q[i+1])
            estimates.append(cosine[i]+t*(cosine[i+1]-cosine[i]))
    return circle, cosine, estimates


def run(output="results"):
    """Save the required graphical plot and sampled crossing estimates."""
    root = output_dirs(output)
    circle, cosine, estimates = graphical_estimates()
    rows = []
    for i, p in enumerate(estimates, 1):
        f = F(p)
        rows.append(dict(intersection=i, x=p[0], y=p[1], f1=f[0], f2=f[1],
                         residual_norm=np.linalg.norm(f), source="plotted-sample interpolation"))
    write_csv(root/"tables/ex2_part_c_graphical_intersections.csv", rows)
    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    ax.plot(*circle.T, label=r"$F_1=0$: $(x-1)^2+y^2=1$")
    ax.plot(*cosine.T, label=r"$F_2=0$: $y=\cos(2x)$")
    for i, p in enumerate(estimates, 1):
        ax.scatter(*p, c="crimson", s=40, zorder=4)
        ax.annotate(f"{i}: ({p[0]:.3f}, {p[1]:.3f})", p,
                    xytext=(-8 if p[0] > 1 else 8, 10),
                    ha="right" if p[0] > 1 else "left", textcoords="offset points", fontsize=8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="$x$", ylabel="$y$", aspect="equal",
           title="2(c): zero curves and graphical intersections")
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    save_figure(fig, root/"figures/ex2_part_c_intersections.png")
    return rows


if __name__ == "__main__":
    module_cli(run)
