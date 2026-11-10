"""Prepare the six report images; preserve every original image and result."""

import csv
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from prepare_report_updates import check_history
from sc_cw1.common.utilities import plt
from sc_cw1.experiments import table
from sc_cw1.exercise1.model import gradient, objective_grid
from sc_cw1.exercise1.part_b_newton import newton
from sc_cw1.exercise1.part_c_minima import distinct_stationary_points
from sc_cw1.exercise1.part_e_comparison import compare as compare_fixed_step
from sc_cw1.exercise2.part_b_contours import grid_minimum_candidates
from sc_cw1.exercise2.part_c_intersections import graphical_estimates
from sc_cw1.exercise2.part_d_gradient_descent import gradient_descent
from sc_cw1.exercise2.part_f_comparison import compare as compare_momentum


def read_table(filename):
    with (PROJECT / "results/tables" / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def newton_contours(points, converged, starts, path):
    """Use the original objective/levels and the fallback-enabled point results."""
    X, Y = np.meshgrid(np.linspace(-2, 2, 501), np.linspace(-2, 2, 501))
    Z = objective_grid(X, Y)
    levels = np.r_[np.linspace(Z.min()+.05, 2, 12), [3, 5, 8, 12, 20, 35, 55]]
    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    contours = ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=.8)
    fig.colorbar(contours, ax=ax, label=r"Objective $J(x_1,x_2)$")
    for point in points:
        if point["lowest_minimum_found"]:
            marker, colour, size, label = "*", "crimson", 140, "Lowest minimum found"
        elif point["classification"] == "minimum":
            marker, colour, size, label = "o", "darkorange", 55, "Other minimum"
        else:
            marker, colour, size, label = "x", "darkorange", 55, "Saddle point"
        xy = (point["x1"], point["x2"])
        ax.scatter(*xy, marker=marker, color=colour, s=size, label=label, zorder=4)
        offset = (7, -13) if label == "Other minimum" else (7, 7)
        ax.annotate(str(point["point_id"]), xy, xytext=offset,
                    textcoords="offset points", fontsize=10)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    ax.set_title(f"1(c): Newton with Armijo; {converged}/{starts} starts converged", fontsize=11)
    ax.legend(loc="lower right", fontsize=10, framealpha=.95)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    """Verify actual data, regenerate only 1(c), and copy five reviewed images."""
    figures = PROJECT / "results/figures"
    destination = figures / "final_report"
    destination.mkdir(exist_ok=True)

    # Explicitly enable fallback: the ordinary 1(c) runner remains strict.
    axis = np.linspace(-2, 2, 21)
    starts = [(u, v) for u in axis for v in axis]
    runs = [newton(x0, tol=1e-6, maxit=200, line_search=True, fallback=True) for x0 in starts]
    points = distinct_stationary_points(runs)
    converged = sum(r["status"] == "converged" for r in runs)
    assert len(starts) == converged == 441
    assert [p["classification"] for p in points] == ["minimum", "minimum", "saddle"]
    assert [p["runs_in_cluster"] for p in points] == [205, 232, 4]
    # The new computation must agree with the separately verified robustness data.
    saved = [p for p in read_table("ex1_newton_robustness_stationary_points.csv")
             if p["method"] == "Safeguarded Newton"]
    for point, previous in zip(points, saved):
        assert np.linalg.norm(gradient((point["x1"], point["x2"]))) < 1e-6
        for key in ("x1", "x2", "value", "gradient_norm", "hessian_determinant", "hessian_11"):
            assert point[key] == float(previous[key])

    fixed = compare_fixed_step()
    for name, result in fixed.items():
        check_history(result, f"ex1_part_e_{name.lower()}_history.csv")
        np.testing.assert_array_equal(result["iterates"][0], [-2, 1])
        assert np.all(result["alphas"][1:] == 1.)
    assert fixed["Newton"]["iterations"] == 5 and fixed["BFGS"]["iterations"] == 12
    assert set(fixed["Newton"]["step_kinds"][1:]) == {"newton"}
    np.testing.assert_array_equal(fixed["BFGS"]["directions"][1], -.01*gradient((-2, 1)))

    minima = grid_minimum_candidates()
    saved_minima = read_table("ex2_part_b_minima.csv")
    np.testing.assert_array_equal(minima, [[float(p["x"]), float(p["y"])] for p in saved_minima])
    assert len(minima) == 2
    _, _, crossings = graphical_estimates()
    saved_crossings = read_table("ex2_part_c_graphical_intersections.csv")
    np.testing.assert_array_equal(crossings, [[float(p["x"]), float(p["y"])] for p in saved_crossings])
    assert len(crossings) == 2

    gd = gradient_descent()
    check_history(gd, "ex2_part_d_gradient_descent_history.csv")
    assert gd["iterations"] == 138 and gd["status"] == "converged"
    np.testing.assert_array_equal(gd["iterates"][0], [0, -1])
    assert np.all(gd["alphas"][1:] == .05) and gd["residual_norms"][-1] < 1e-6
    assert np.all(gd["iterates"] > [-.1, -1.15]) and np.all(gd["iterates"] < [1.6, .15])
    momentum = compare_momentum()
    assert [r["iterations"] for r in momentum.values()] == [64, 36, 79]
    for gamma, result in momentum.items():
        check_history(result, f"ex2_part_f_gamma_{gamma:.1f}_history.csv")
        assert result["status"] == "converged"

    new_name = "ex1_part_c_contours.png"
    reused = ["ex1_part_e_newton_vs_bfgs.png", "ex2_part_b_residual_contours.png",
              "ex2_part_c_intersections.png", "ex2_part_d_gradient_descent.png",
              "ex2_part_f_momentum_subplots.png"]
    newton_contours(points, converged, len(starts), destination / new_name)
    for name in reused:
        shutil.copy2(figures / name, destination / name)
        assert (figures / name).read_bytes() == (destination / name).read_bytes()
    assert {p.name for p in destination.iterdir()} == {new_name, *reused}

    sources = ["scripts/prepare_final_figures.py:newton_contours; actual newton(..., fallback=True)",
               "results/figures/ex1_part_e_newton_vs_bfgs.png; exercise1/part_e_comparison.py:run",
               "results/figures/ex2_part_b_residual_contours.png; exercise2/part_b_contours.py:run",
               "results/figures/ex2_part_c_intersections.png; exercise2/part_c_intersections.py:run",
               "results/figures/ex2_part_d_gradient_descent.png; exercise2/part_d_gradient_descent.py:run",
               "results/figures/ex2_part_f_momentum_subplots.png; scripts/prepare_report_updates.py:momentum_panels"]
    manifest = []
    for name, source in zip([new_name]+reused, sources):
        manifest.append(dict(figure=f"[{name}](results/figures/final_report/{name})",
                             action="Newly generated, 600 DPI" if name == new_name else "Copied byte-for-byte",
                             source=source))
    minima_text = ", ".join(f"({x:.2f}, {y:.2f})" for x, y in minima)
    notes = f"""# Final report figure manifest

The six final images are in `scientific-computation-cw1/results/figures/final_report/`.
Run `python scripts/prepare_final_figures.py` from the project directory to
reproduce this selection. The script explicitly enables fallback only for the
new 1(c) image, recomputes the relevant verification data with the original
solvers, and copies the five reviewed originals unchanged. It does not call
the default strict-Newton 1(c) plotting runner or overwrite original outputs.

## Final images and provenance

Paths in the links are relative to `scientific-computation-cw1/`.
Package source paths in the last column are relative to `src/sc_cw1/` unless
prefixed with `scripts/` or `results/`.

{table(manifest, ['figure', 'action', 'source'])}

The copied originals have PNG resolution metadata of 300 DPI for 1(e), 2(b),
2(c), 2(d), and 600 DPI for the momentum panels. The new 1(c) image is 600 DPI.
Use approximately full text width for the original plot labels and the three
momentum panels. Images and originals were visually reviewed for readable
titles, axes, legends, contour colour bars and unclipped trajectories/markers.

## Numerical verification

- **1(c):** {len(starts)} starts on the unchanged 21×21 square grid; all
  {converged} converge with Armijo and fallback enabled, tol=1e-6, maxit=200.
  The three computed clusters have counts
  {', '.join(str(p['runs_in_cluster']) for p in points)}. Markers use computed
  coordinates and the existing gradient/Hessian classification. The contour
  domain is [-2,2]² and its colour bar is the original objective J.

{table(points, ['point_id', 'classification', 'x1', 'x2', 'value', 'runs_in_cluster'])}

- **1(e):** initial point (-2,1), alpha=1, B0=.01 I; pure fixed-step Newton
  {fixed['Newton']['iterations']} updates and BFGS {fixed['BFGS']['iterations']}.
  All accepted Newton steps are labelled `newton`; no fallback is used. Both
  norm histories, unit steps and the 1e-6 tolerance match the semilog figure.
- **2(b):** phi=F1²+F2² on [-2,2]². The two apparent grid minima are
  {minima_text}, plotted to two decimal places. These finite-grid locations
  are not labelled as exact roots.
- **2(c):** the unit circle centred at (1,0) and y=cos(2x) give graphical
  estimates ({crossings[0][0]:.7f}, {crossings[0][1]:.7f}) and
  ({crossings[1][0]:.7f}, {crossings[1][1]:.7f}), consistent with the displayed
  three-decimal labels. Only the original sampled-curve procedure is used.
- **2(d):** start (0,-1), step .05, ||F||₂<1e-6, maxit=200;
  {gd['iterations']} accepted updates. The complete history matches the CSV
  and lies inside the original plotted limits, including start/final markers.
- **2(f):** gamma=.1/.3/.7 gives
  {'/'.join(str(r['iterations']) for r in momentum.values())} accepted updates.
  The original panels use one shared padded domain and the same 16 phi levels
  from 1e-5 to 10. They show all actual trajectories, starts and final points.

Every recomputed fixed-step/descent/momentum history is checked against every
row of the original CSV. The recomputed 1(c) points agree with the verified
fallback-enabled robustness table. No solver, starting grid, stopping rule,
numerical table or original figure is changed. The lowest minimum remains
numerical evidence, not a formal global-optimality proof.

## Import into Overleaf

Download these six PNG files from `results/figures/final_report/` and upload
them into a folder named `figures` in Overleaf, keeping their filenames.
Use the final-report versions rather than the original 1(c) image. Do not
import the two-panel Newton comparison or the old overlaid momentum figure.
For example:

```latex
\\includegraphics[width=\\linewidth]{{figures/ex1_part_c_contours.png}}
```

The repository's normal `scripts/run_all.py` continues to produce the original
strict-Newton experiment. Use the dedicated final-figure command above for
this report selection. No LaTeX file, PDF, References section or Generative AI
appendix is edited or generated by this script.
"""
    (PROJECT / "FINAL_FIGURES_NOTES.md").write_text(notes, encoding="utf-8")
    print(f"Verified six final report figures in {destination}")
    print(f"1(c): {converged}/{len(starts)} converged; clusters {[p['runs_in_cluster'] for p in points]}")
    print("Reused five originals byte-for-byte; no original result files overwritten")


if __name__ == "__main__":
    main()
