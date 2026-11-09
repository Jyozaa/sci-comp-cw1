"""Add report evidence without overwriting the original coursework outputs."""

import ast
import csv
from pathlib import Path
import sys
import textwrap

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sc_cw1.common.utilities import plt, write_csv
from sc_cw1.experiments import table
from sc_cw1.exercise1.part_e_comparison import compare as compare_fixed_step
from sc_cw1.exercise2.model import phi_grid
from sc_cw1.exercise2.part_d_gradient_descent import gradient_descent
from sc_cw1.exercise2.part_f_comparison import compare as compare_momentum


def check_history(result, filename):
    """Check the recomputed run against every row of its existing CSV history."""
    with (PROJECT / "results/tables" / filename).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    points = np.array([[float(row["x1"]), float(row["x2"])] for row in rows])
    np.testing.assert_array_equal(points, result["iterates"])
    np.testing.assert_array_equal([int(row["k"]) for row in rows], np.arange(len(rows)))
    for column, key in [("value", "values"), ("gradient_norm", "gradient_norms"),
                        ("alpha", "alphas"), ("residual_norm", "residual_norms")]:
        if key in result:
            np.testing.assert_array_equal([float(row[column]) for row in rows], result[key])
    assert [row["step_kind"] for row in rows] == result["step_kinds"]
    assert all(row["status"] == result["status"] for row in rows)
    assert result["iterations"] == len(rows)-1


def momentum_panels(runs, path):
    """Plot each original trajectory on the same padded coordinate domain."""
    points = np.concatenate([r["iterates"] for r in runs.values()])
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = .08*(upper-lower)
    xlim = (lower[0]-padding[0], upper[0]+padding[0])
    ylim = (lower[1]-padding[1], upper[1]+padding[1])
    X, Y = np.meshgrid(np.linspace(*xlim, 350), np.linspace(*ylim, 350))
    Z = phi_grid(X, Y)
    levels = np.geomspace(1e-5, 10, 16)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.99, bottom=.16, top=.86, wspace=.18)
    for ax, (gamma, result), colour in zip(axes, runs.items(), ["C0", "C1", "C2"]):
        xy = result["iterates"]
        assert np.all(xy > lower-padding) and np.all(xy < upper+padding)
        ax.contour(X, Y, Z, levels=levels, colors=".82", linewidths=.5)
        ax.plot(xy[:, 0], xy[:, 1], "o-", color=colour, linewidth=1.3, markersize=2.3)
        ax.scatter(*xy[0], marker="s", color="black", s=35, zorder=4, label="Initial point")
        ax.scatter(*xy[-1], marker="*", color="black", s=95, zorder=5, label="Converged point")
        ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="$x$")
        ax.set_title(rf"$\gamma={gamma}$"+f"\n{result['iterations']} updates", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.locator_params(axis="x", nbins=3)
    axes[0].set_ylabel("$y$", fontsize=11)
    fig.suptitle(r"Momentum trajectories, $\lambda=0.05$", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=11, frameon=False)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return xlim, ylim


def momentum_excerpt():
    """Extract one continuous function-body excerpt, including the safeguards."""
    path = PROJECT / "src/sc_cw1/exercise2/part_d_gradient_descent.py"
    source = path.read_text()
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "gradient_descent")
    first = next(n for n in function.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == "k")
    last = function.body[-2]  # Final convergence check; omit only the return below it.
    lines = source.splitlines()[first.lineno-1:last.end_lineno]
    return textwrap.dedent("\n".join(lines))


def main():
    """Reproduce only the requested additions using the original experiments."""
    figures = PROJECT / "results/figures"
    tables = PROJECT / "results/tables"
    gd = gradient_descent()
    fixed = compare_fixed_step()
    momentum = compare_momentum()
    check_history(gd, "ex2_part_d_gradient_descent_history.csv")
    for name, result in fixed.items():
        check_history(result, f"ex1_part_e_{name.lower()}_history.csv")
    for gamma, result in momentum.items():
        check_history(result, f"ex2_part_f_gamma_{gamma:.1f}_history.csv")
        assert result["status"] == "converged"

    gd_rows = [dict(k=k, x=gd["iterates"][k, 0], y=gd["iterates"][k, 1],
                    residual_norm=gd["residual_norms"][k], gradient_norm=gd["gradient_norms"][k])
               for k in (0, 10, 40, 80, 138)]
    fixed_rows = []
    for k in (0, 1, 3, 5, 8, 12):
        row = dict(k=k)
        for name, result in fixed.items():
            row[name] = result["gradient_norms"][k] if k <= result["iterations"] else ""
        fixed_rows.append(row)
    write_csv(tables / "ex2_part_d_selected_iterations.csv", gd_rows)
    write_csv(tables / "ex1_part_e_selected_gradient_norms.csv", fixed_rows)
    figure = figures / "ex2_part_f_momentum_subplots.png"
    xlim, ylim = momentum_panels(momentum, figure)

    ratios = gd["residual_norms"][1:]/gd["residual_norms"][:-1]
    excerpt = momentum_excerpt()
    notes = f"""# CW1 report update notes

Reproduce these additions from `scientific-computation-cw1/` with
`python scripts/prepare_report_updates.py` using the existing environment.
The script calls the original solvers with their original parameters and checks
every returned history row against the existing CSVs before writing additions.
It does not overwrite any original figure, numerical result or report file.

## Exercise 2(f): separate momentum trajectories

New 600 DPI figure:
[results/figures/ex2_part_f_momentum_subplots.png](results/figures/ex2_part_f_momentum_subplots.png).
Paths here are relative to `scientific-computation-cw1/`.
All panels use the same phi contours (16 levels from 1e-5 to 10) and limits
derived from every trajectory point, with 8% padding:
x in [{xlim[0]:.6f}, {xlim[1]:.6f}], y in [{ylim[0]:.6f}, {ylim[1]:.6f}].
Black squares mark the common start (0,-1); black stars mark the final points.
Each title gives the actual update count. All runs converge using step .05,
residual tolerance 1e-6 and at most 200 updates, with fresh zero momentum.
The original overlaid figure remains available at
[results/figures/ex2_part_f_momentum_comparison.png](results/figures/ex2_part_f_momentum_comparison.png).
Use the new figure at approximately full text width; its fonts are 10–12 points
at the saved 7.2-inch figure width, with one shared legend.

## Exercise 2(d): selected ordinary-descent iterates

CSV: [results/tables/ex2_part_d_selected_iterations.csv](results/tables/ex2_part_d_selected_iterations.csv).
The residual column is ||F||₂; the gradient column is ||2 J_F.T F||₂.
The start is (0,-1), step .05, stop ||F||₂<1e-6, and maxit=200.
k counts accepted updates; k=0 is the initial point.

{table(gd_rows, ['k', 'x', 'y', 'residual_norm', 'gradient_norm'])}

- From k=0 to k=10 the residual falls from {gd_rows[0]['residual_norm']:.6g}
  to {gd_rows[1]['residual_norm']:.6g}, while the gradient norm falls from
  {gd_rows[0]['gradient_norm']:.6g} to {gd_rows[1]['gradient_norm']:.6g}.
  Thus a relatively small gradient occurs while the residual is still about .94.
  On the update ending at k=10 the residual ratio is {ratios[9]:.9f}, a reduction
  of only {100*(1-ratios[9]):.3f}%. Small residual-gradient norm alone does not
  establish a root; these data do not classify that region as a saddle.
- Near the final root, both norms decrease, but the absolute changes become
  small. The per-update residual ratios at k=40, 80 and 138 are respectively
  {ratios[39]:.9f}, {ratios[79]:.9f} and {ratios[137]:.9f}. The later ratios are
  closer to one, so fractional progress slows somewhat relative to k=40.
- The last-step factor remains {ratios[-1]:.9f}, approximately .8982: about
  {100*(1-ratios[-1]):.2f}% residual reduction on that update. This is an
  observed history ratio, not a convergence-order theorem or eigenvalue analysis.

## Exercise 1(e): selected gradient norms

CSV: [results/tables/ex1_part_e_selected_gradient_norms.csv](results/tables/ex1_part_e_selected_gradient_norms.csv).
Both columns give ||grad J(x_k)||₂. Settings: x0=(-2,1), fixed alpha=1,
Armijo disabled, BFGS B0=.01 I, gradient tolerance 1e-6, maxit=200.
Empty cells mean that the algorithm had already terminated; final values
are not repeated at later iteration numbers.

{table(fixed_rows, ['k', 'Newton', 'BFGS'])}

Newton takes {fixed['Newton']['iterations']} updates; BFGS takes {fixed['BFGS']['iterations']}.
Newton's norm falls from {fixed['Newton']['gradient_norms'][3]:.6g} at k=3 to
{fixed['Newton']['gradient_norms'][-1]:.6g} at its final update. BFGS makes slower
initial progress here: at k=3 its norm is {fixed['BFGS']['gradient_norms'][3]:.6g}.
BFGS starts with a scalar inverse-Hessian approximation .01 I and adjusts it
using successive s/y pairs, whereas Newton uses the current Hessian. This
explains the different directions and is consistent with the early histories.
A Newton iteration evaluates the Hessian and solves H*delta=-g; BFGS uses
matrix-vector/outer products and its inverse update, without a Hessian solve.
Both evaluate the objective and gradient at new points. Counts alone do not
measure runtime, and this comparison concerns only the prescribed start,
scaling and fixed steps; it is not a general ranking or convergence theorem.

## Exercise 2(e): one coherent momentum excerpt

Source: `src/sc_cw1/exercise2/part_d_gradient_descent.py:gradient_descent`.
The following is one exact continuous function-body excerpt, dedented only.
The surrounding imports, function signature, parameter validation and final
return dictionary are omitted; all initialisation, numerical safeguards and
iteration recording within the excerpt are retained. NumPy is `np`;
`residual` is the existing F, `jac` its analytical Jacobian. Parameters are
x0=(0,-1), step_size=.05, tol=1e-6 and maxit=200. The momentum wrapper supplies
gamma; gamma=0 gives ordinary gradient descent in this very same loop.

```python
{excerpt}
```

The return below this excerpt stores the accepted iterates/norms, status and
iterations=k. The three marked momentum changes initialise d_prev, add its
weighted value to g, and retain the accepted direction. The recurrence and
residual stopping test are exactly those executed in the coursework runs.

## Qualifications for the separate LaTeX revision

Insert the new figure, select the two compact evidence tables, and replace the
fragmented momentum listings with the excerpt as space permits. Tables retain
full precision in CSV; Markdown display rounds to nine significant digits.
The initial point is not an update. No new optimisation method, numerical
study, eigenvalue calculation or formal global-minimum proof is supplied here.
Existing limits on global-minimum justification still apply. No LaTeX, PDF,
References or Generative AI appendix was edited.
"""
    (PROJECT / "REPORT_UPDATE_NOTES.md").write_text(notes, encoding="utf-8")
    print(f"New figure: {figure}")
    print(f"Evidence tables: {tables / 'ex2_part_d_selected_iterations.csv'}")
    print(f"                 {tables / 'ex1_part_e_selected_gradient_norms.csv'}")
    print(f"Notes: {PROJECT / 'REPORT_UPDATE_NOTES.md'}")


if __name__ == "__main__":
    main()
