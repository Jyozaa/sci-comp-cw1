"""2(f): independent trajectories for the three prescribed momentum constants."""

import numpy as np
from sc_cw1.common.utilities import output_dirs, pyplot, plot_path, save_figure, save_history, summary, write_csv, module_cli
from .part_e_momentum import momentum_descent
from .part_d_gradient_descent import add_residual_background


def compare():
    """Fresh x0=(0,-1), d_-1=0 for each gamma in (.1,.3,.7), lambda=.05."""
    return {g: momentum_descent(gamma=g, step_size=.05, tol=1e-6, maxit=200)
            for g in (.1, .3, .7)}


def trajectory_statistics(result):
    """Measure objective increases, reversed consecutive steps, and path length.

    Negative dot products between successive displacement vectors flag reversals;
    this is an observable oscillation diagnostic, not a stability proof.
    """
    displacement = np.diff(result.iterates, axis=0)
    values = np.array([r["value"] for r in result.history])
    return dict(objective_increases=int(np.sum(np.diff(values) > 0)),
                step_reversals=int(np.sum(np.sum(displacement[1:]*displacement[:-1], axis=1) < 0)),
                path_length=float(np.linalg.norm(displacement, axis=1).sum()))


def run(output="results"):
    """Save the momentum trajectory comparison and all three full histories."""
    root = output_dirs(output)
    runs = compare()
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
    for ax in axes:
        add_residual_background(ax, xlim=(-.1, 1.6), ylim=(-1.15, 1.15))
    for (gamma, result), color, marker in zip(runs.items(), ["C0", "C1", "C2"], ["o", "^", "D"]):
        save_history(root / "tables" / f"ex2_part_f_gamma_{gamma:.1f}_history.csv", result)
        for ax in axes:
            plot_path(ax, result, f"gamma={gamma}: {result.status}, {result.nit} steps", color, marker)
    axes[0].set_title("2(f): momentum trajectories; step = 0.05")
    axes[1].set(xlim=(.2, .5), ylim=(.6, .9), title="Detail near approached solution")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), fontsize=8, loc="outside lower center", ncol=3)
    save_figure(fig, root / "figures/ex2_part_f_momentum_comparison.png")
    write_csv(root / "tables/ex2_part_f_summary.csv",
              [dict(summary(r), **trajectory_statistics(r)) for r in runs.values()])
    return runs


if __name__ == "__main__":
    module_cli(run)
