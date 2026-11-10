"""Compare opt-in safeguarded Newton with strict Newton; write only new outputs."""

import ast
from collections import Counter
from pathlib import Path
import sys
import textwrap

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sc_cw1.common.utilities import plt, write_csv
from sc_cw1.experiments import table
from sc_cw1.exercise1.model import objective, gradient, hessian, objective_grid
from sc_cw1.exercise1.part_b_newton import newton
from sc_cw1.exercise1.part_c_minima import multistart, distinct_stationary_points


def check_steps(result):
    """Check recorded directions and Armijo acceptance against the actual model."""
    assert result["iterations"] == len(result["iterates"])-1
    for k in range(1, len(result["iterates"])):
        x, delta = result["iterates"][k-1], result["directions"][k]
        g, alpha = gradient(x), result["alphas"][k]
        assert np.all(np.isfinite(delta)) and g @ delta < 0
        if result["step_kinds"][k] == "gradient_fallback":
            np.testing.assert_array_equal(delta, -g)
        else:
            assert result["step_kinds"][k] == "newton"
            np.testing.assert_allclose(hessian(x) @ delta, -g, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(result["iterates"][k], x+alpha*delta)
        assert result["values"][k] <= result["values"][k-1]+1e-4*alpha*(g @ delta)


def plot_start_destinations(rows, summary, path):
    """Show sampled starting-point outcomes, with the original J as background."""
    X, Y = np.meshgrid(np.linspace(-2, 2, 301), np.linspace(-2, 2, 301))
    Z = objective_grid(X, Y)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.99, bottom=.23, top=.84, wspace=.16)
    for ax, totals in zip(axes, summary):
        selected = [r for r in rows if r["method"] == totals["method"]]
        ax.contour(X, Y, Z, levels=[-1.4, -.5, 0, 1, 2, 5, 10, 20, 40],
                   colors=".85", linewidths=.4)
        ids = sorted({r["point_id"] for r in selected if r["point_id"] != ""})
        for point_id in ids:
            group = [r for r in selected if r["point_id"] == point_id]
            label = f"P{point_id}: {group[0]['classification']}"
            marker = "^" if group[0]["classification"] == "saddle" else "o"
            ax.scatter([r["start_x1"] for r in group], [r["start_x2"] for r in group],
                       s=15, marker=marker, color=f"C{point_id-1}", label=label, zorder=3)
        failed = [r for r in selected if r["status"] != "converged"]
        if failed:
            ax.scatter([r["start_x1"] for r in failed], [r["start_x2"] for r in failed],
                       marker="x", s=15, linewidths=.7, color=".45", label="Unsuccessful", zorder=3)
        ax.set(xlim=(-2.12, 2.12), ylim=(-2.12, 2.12), aspect="equal", xlabel="$x_1$")
        ax.set_title(f"{totals['method']}\n{totals['converged']}/{totals['starts']} converged", fontsize=11)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("$x_2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle("Newton/Armijo: sampled starting-point destinations", fontsize=12)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run strict first, then safeguarded Newton, on the unchanged 21×21 grid."""
    starts, strict, strict_points = multistart()
    safeguarded = [newton(x0, fallback=True) for x0 in starts]
    safeguarded_points = distinct_stationary_points(safeguarded)
    experiments = [("Strict Newton", strict, strict_points),
                   ("Safeguarded Newton", safeguarded, safeguarded_points)]
    summary, distribution, rows = [], [], []
    for method, runs, points in experiments:
        for point in points:
            assert np.linalg.norm(gradient((point["x1"], point["x2"]))) < 1e-6
            distribution.append(dict(method=method, **point))
        fallback_counts = []
        for x0, result in zip(starts, runs):
            check_steps(result)
            count = result["step_kinds"].count("gradient_fallback")
            fallback_counts.append(count)
            point_id, classification, reason = "", "not_classified", ""
            if result["status"] == "converged":
                point = next(p for p in points
                             if np.linalg.norm(result["x"]-np.array([p["x1"], p["x2"]])) < 1e-4)
                point_id, classification = point["point_id"], point["classification"]
            else:
                reason = result["status"]
                if method == "Strict Newton" and reason == "line_search_failure":
                    g = gradient(result["x"])
                    delta = np.linalg.solve(hessian(result["x"]), -g)
                    reason = "non_descent_newton_direction" if g @ delta >= 0 else "armijo_search_failed"
            newton_count = result["step_kinds"].count("newton")
            assert newton_count+count == result["iterations"]
            rows.append(dict(method=method, start_x1=x0[0], start_x2=x0[1], status=result["status"],
                             iterations=result["iterations"], x1=result["x"][0], x2=result["x"][1],
                             value=result["values"][-1], gradient_norm=result["gradient_norms"][-1],
                             newton_steps=newton_count, fallback_steps=count, point_id=point_id,
                             classification=classification, failure_cause=reason))
        iterations = [r["iterations"] for r in runs if r["status"] == "converged"]
        summary.append(dict(method=method, starts=len(starts), converged=len(iterations),
                            failed=len(starts)-len(iterations), runs_with_fallback=sum(c > 0 for c in fallback_counts),
                            accepted_fallback_steps=sum(fallback_counts), mean_converged_updates=np.mean(iterations),
                            max_converged_updates=max(iterations)))

    tables = PROJECT / "results/tables"
    write_csv(tables / "ex1_newton_robustness_summary.csv", summary)
    write_csv(tables / "ex1_newton_robustness_stationary_points.csv", distribution)
    write_csv(tables / "ex1_newton_robustness_runs.csv", rows)
    figure = PROJECT / "results/figures/ex1_newton_robustness_basins.png"
    plot_start_destinations(rows, summary, figure)

    origin = [("Strict Newton", newton((0, 0))),
              ("Safeguarded Newton", newton((0, 0), fallback=True))]
    origin_rows = []
    for method, result in origin:
        check_steps(result)
        for k, x in enumerate(result["iterates"]):
            origin_rows.append(dict(method=method, k=k, x1=x[0], x2=x[1], value=result["values"][k],
                                    gradient_norm=result["gradient_norms"][k], alpha=result["alphas"][k],
                                    step_kind=result["step_kinds"][k], direction1=result["directions"][k, 0],
                                    direction2=result["directions"][k, 1], status=result["status"]))
    write_csv(tables / "ex1_newton_origin_history.csv", origin_rows)
    g0 = gradient((0, 0))
    H0 = hessian((0, 0))
    delta0 = np.linalg.solve(H0, -g0)
    safe_origin = origin[1][1]
    causes = {method: dict(Counter(r["failure_cause"] for r in rows
                                  if r["method"] == method and r["status"] != "converged"))
              for method, _, _ in experiments}
    failure_text = []
    for method, counts in causes.items():
        description = ", ".join(f"{count} {cause.replace('_', ' ')}" for cause, count in counts.items())
        failure_text.append(f"- {method}: {description or 'no failures'}.")
    failure_text = "\n".join(failure_text)
    source = (PROJECT / "src/sc_cw1/exercise1/part_b_newton.py").read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "newton")
    # Extract the actual direction-selection block, rather than a replacement solver.
    lines = source.splitlines()
    first = next(i for i in range(function.lineno-1, function.end_lineno) if lines[i].strip() == "H = hess(x)")
    last = next(i for i in range(first, function.end_lineno) if lines[i].strip() == "x_new = x+alpha*delta")
    excerpt = textwrap.dedent("\n".join(lines[first:last+1]))
    notes = f"""# Newton robustness report notes

Reproduce only these additions from `scientific-computation-cw1/` with
`python scripts/compare_newton_robustness.py`. The original 1(b/c) experiment
remains strict by default; the original output files are preserved.

## Modification and scope

`newton(..., fallback=False)` is the unchanged strict default. With
`fallback=True` **and** Armijo enabled, the usual H*delta=-g solve is attempted.
An unusable Hessian/solve, nonfinite direction or non-descent direction selects
delta=-g. If backtracking fails for a usable Newton direction, Armijo is retried
once with -g. A negative-gradient search is never retried a second time.
Every accepted step must pass the original objective Armijo condition.

For nonzero g, g.T*(-g)=-||g||₂²<0, giving a descent direction. This is a
safeguard using taught steepest descent, not a modification of the Newton
equation. `line_search=False` always stays pure Newton, even if fallback=True.
The result schema is unchanged; accepted `step_kinds` distinguish `newton`
and `gradient_fallback`. Counts below refer to accepted steps, not attempts.

## Same-grid comparison

Settings: 21×21 uniformly spaced starts in [-2,2]²; ||grad J||₂<1e-6;
maxit=200; Armijo initially alpha=1, c1=1e-4, halving, up to 50 backtracks.

{table(summary, ['method', 'starts', 'converged', 'failed', 'runs_with_fallback', 'accepted_fallback_steps'])}

{table(summary, ['method', 'mean_converged_updates', 'max_converged_updates'])}

The means and maxima include only converged runs. Their populations differ,
so they do not measure efficiency on identical sets of starts. Actual failure
causes are:

{failure_text}

## Stationary-point distribution and checks

{table(distribution, ['method', 'point_id', 'classification', 'x1', 'x2', 'value', 'runs_in_cluster'])}

The existing distance grouping and analytical-gradient/Hessian classification
are used. Full derivative norms, Hessian determinants and H11 appear in the
stationary-point CSV. Only converged runs are grouped. Both variants identify
the same two minima and saddle to the stopping/grouping tolerances. P1 is the
lowest minimum found numerically, not a proven global minimum.
The {sum(p['runs_in_cluster'] for p in safeguarded_points if p['classification'] == 'saddle')} safeguarded saddle
outcomes remain possible because the stopping test recognises stationarity,
which does not imply a minimum.

## Illustrative start (0,0)

At this point, J={objective((0, 0)):.9g}, g={g0.tolist()} and H={H0.tolist()}.
The computed Newton direction is {delta0.tolist()}, with g.T*delta={g0 @ delta0:.9g}:
it fails the descent test, so strict Newton terminates with
`{origin[0][1]['status']}` after {origin[0][1]['iterations']} accepted updates.

The safeguarded run selects {safe_origin['directions'][1].tolist()}, whose
gradient-direction product is {g0 @ safe_origin['directions'][1]:.9g}. Armijo accepts
alpha={safe_origin['alphas'][1]:.9g}, reaching {safe_origin['iterates'][1].tolist()}.
It finishes `{safe_origin['status']}` after {safe_origin['iterations']} updates,
with {safe_origin['step_kinds'].count('newton')} Newton and
{safe_origin['step_kinds'].count('gradient_fallback')} fallback steps.
The final point is ({safe_origin['x'][0]:.9g}, {safe_origin['x'][1]:.9g}), with
J={safe_origin['values'][-1]:.12g} and ||grad J||₂={safe_origin['gradient_norms'][-1]:.9g}.

{table(origin_rows, ['method', 'k', 'x1', 'x2', 'value', 'gradient_norm', 'alpha', 'step_kind'])}

The origin history includes every accepted direction in CSV. An Armijo decrease
in J does not require the gradient norm to decrease on every step.

## Exact direction-selection source excerpt

Source: `src/sc_cw1/exercise1/part_b_newton.py:newton`. This is one continuous
excerpt from inside the main while loop, dedented only. The surrounding code
has evaluated g, checked the gradient tolerance and initial finiteness;
trial objective/gradient evaluation, finite-trial
checks and history recording follow it. Accepted updates append `step_type`
to `step_types` (returned as `step_kinds`) and only then increment k.

```python
{excerpt}
```

## New output locations

All paths are relative to `scientific-computation-cw1/`:

- [Summary](results/tables/ex1_newton_robustness_summary.csv)
- [Stationary points and analytical checks](results/tables/ex1_newton_robustness_stationary_points.csv)
- [All {len(rows)} starting-point outcomes and accepted step counts](results/tables/ex1_newton_robustness_runs.csv)
- [Complete origin histories](results/tables/ex1_newton_origin_history.csv)
- [Sampled starting-point comparison](results/figures/ex1_newton_robustness_basins.png)

The figure plots **initial guesses**, coloured by the stationary point reached;
unsuccessful strict starts are grey crosses. Both panels have identical scales
and original-J contours. It depicts sampled destinations, not analytically
established basin boundaries. Original figures are not overwritten.

## Benefits, limitations and later report wording

The safeguard improves convergence coverage on this grid while explicitly
recording when another direction is used. These runs are safeguarded Newton,
not pure Newton on every step. Finite tests do not guarantee convergence for
arbitrary starts or establish global optimality. Numerical failure and search
exhaustion remain possible; failures are never classified as minima.
The existing one-variable/global-minimum discussion is unchanged.

For the separate 1(b/c) revision, retain the strict results as the baseline,
describe the safeguard as optional, include accepted fallback counts, and
distinguish objective descent from convergence to a minimum. Keep 1(e)'s
fixed-step Newton/BFGS comparison separate and unchanged. No LaTeX, PDF,
References or Generative AI appendix is edited by this script.
"""
    (PROJECT / "NEWTON_ROBUSTNESS_REPORT_NOTES.md").write_text(notes, encoding="utf-8")
    print(table(summary, list(summary[0])))
    print(f"New files: {tables}/ex1_newton_robustness_*.csv; ex1_newton_origin_history.csv")
    print(f"Figure: {figure}")
    print(f"Notes: {PROJECT / 'NEWTON_ROBUSTNESS_REPORT_NOTES.md'}")


if __name__ == "__main__":
    main()
