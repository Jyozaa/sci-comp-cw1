"""1(e): prescribed unit-step Newton versus BFGS comparison."""

from .part_b_newton import newton
from .part_d_bfgs import bfgs
from sc_cw1.common.utilities import output_dirs, plt, save_figure, save_history, module_cli


def compare():
    """Both use x0=(-2,1), alpha=1; BFGS uses B0=.01 I."""
    return {"Newton": newton((-2, 1), line_search=False, fallback=False),
            "BFGS": bfgs((-2, 1), omega=.01, line_search=False)}


def run(output="results"):
    """Save the required gradient-norm comparison and both complete histories."""
    root = output_dirs(output)
    runs = compare()
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    for name, r in runs.items():
        save_history(root/"tables"/f"ex1_part_e_{name.lower()}_history.csv", r)
        ax.semilogy(range(len(r["iterates"])), r["gradient_norms"], marker="o", markersize=3,
                    label=f"{name}: {r['status']} ({r['iterations']} updates)")
    ax.axhline(1e-6, color="grey", linestyle=":", label="Tolerance")
    ax.set(xlabel="Iteration k (including the initial guess)", ylabel=r"$\|\nabla J(x_k)\|_2$",
           title="1(e): fixed unit steps; start (-2,1); BFGS omega=0.01")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)
    save_figure(fig, root/"figures/ex1_part_e_newton_vs_bfgs.png")
    return runs


if __name__ == "__main__":
    module_cli(run)
