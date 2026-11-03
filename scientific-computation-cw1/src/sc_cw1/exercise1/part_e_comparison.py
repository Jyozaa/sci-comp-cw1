"""1(e): prescribed unit steps, without line search or Newton fallback."""

from sc_cw1.common.utilities import output_dirs, pyplot, save_figure, save_history, summary, write_csv, module_cli
from .part_b_newton import newton
from .part_d_bfgs import bfgs


def compare(*, tol=1e-6, maxit=200):
    """Run both algorithms at x0=(-2,1), alpha=1, B0=0.01 I for BFGS.

    tol and maxit are supplementary choices for Exercise 1, not printed values.
    Unsafe curvature causes a reported failure, never a modified BFGS update.
    """
    return {"Newton": newton((-2, 1), step_mode="fixed", alpha=1., fallback="none", tol=tol, maxit=maxit),
            "BFGS": bfgs((-2, 1), step_mode="fixed", alpha=1., omega=.01,
                         curvature_action="fail", tol=tol, maxit=maxit)}


def run(output="results"):
    """Save 1(e) gradient histories, outcomes and a semilog convergence figure."""
    root = output_dirs(output)
    runs = compare()
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    for name, r in runs.items():
        save_history(root / "tables" / f"ex1_part_e_{name.lower()}_history.csv", r)
        ax.semilogy([row["k"] for row in r.history], [row["gradient_norm"] for row in r.history],
                    marker="o", markersize=3, label=f"{name}: {r.status} ({r.nit} steps)")
    ax.axhline(1e-6, color="grey", linestyle=":", label="Tolerance")
    ax.set(xlabel="Iteration k (including initial guess)", ylabel=r"$\|\nabla J(x_k)\|_2$",
           title=r"1(e): fixed unit steps, $x_0=(-2,1)$; BFGS $\omega=0.01$")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=.25)
    save_figure(fig, root / "figures/ex1_part_e_newton_vs_bfgs.png")
    write_csv(root / "tables/ex1_part_e_summary.csv", [summary(r, method=n) for n, r in runs.items()])
    return runs


if __name__ == "__main__":
    module_cli(run)
