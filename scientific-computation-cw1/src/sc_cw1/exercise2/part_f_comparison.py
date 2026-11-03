"""2(f): compare the three prescribed momentum coefficients."""

from .part_e_momentum import momentum_descent
from .part_d_gradient_descent import residual_background
from sc_cw1.common.utilities import output_dirs, plt, save_figure, save_history, module_cli


def compare():
    """Each call starts at (0,-1), with a fresh zero previous direction."""
    return {gamma: momentum_descent(gamma=gamma) for gamma in (.1, .3, .7)}


def run(output="results"):
    """Save one clear trajectory comparison and the three histories."""
    root = output_dirs(output)
    runs = compare()
    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    residual_background(ax, xlim=(-.1, 1.15), ylim=(-1.15, 1.15))
    for (gamma, r), marker in zip(runs.items(), ["o", "^", "D"]):
        path = r["iterates"]
        ax.plot(path[:, 0], path[:, 1], marker=marker, markersize=2.5, linewidth=1,
                label=f"gamma={gamma}: {r['iterations']} updates, {r['status']}")
        save_history(root/"tables"/f"ex2_part_f_gamma_{gamma:.1f}_history.csv", r)
    ax.scatter(0, -1, marker="s", color="black", label="Start", zorder=5)
    # The three final points coincide to the accuracy visible in this plot.
    for r in runs.values():
        ax.scatter(*r["x"], marker="*", s=100, color="black", zorder=5)
    ax.scatter([], [], marker="*", color="black", label="Final points")
    ax.set_title("2(f): momentum trajectories; fixed step 0.05")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.11))
    save_figure(fig, root/"figures/ex2_part_f_momentum_comparison.png")
    return runs


if __name__ == "__main__":
    module_cli(run)
