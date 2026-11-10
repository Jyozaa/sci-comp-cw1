# Final report figure manifest

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

| figure | action | source |
| --- | --- | --- |
| [ex1_part_c_contours.png](results/figures/final_report/ex1_part_c_contours.png) | Newly generated, 600 DPI | scripts/prepare_final_figures.py:newton_contours; actual newton(..., fallback=True) |
| [ex1_part_e_newton_vs_bfgs.png](results/figures/final_report/ex1_part_e_newton_vs_bfgs.png) | Copied byte-for-byte | results/figures/ex1_part_e_newton_vs_bfgs.png; exercise1/part_e_comparison.py:run |
| [ex2_part_b_residual_contours.png](results/figures/final_report/ex2_part_b_residual_contours.png) | Copied byte-for-byte | results/figures/ex2_part_b_residual_contours.png; exercise2/part_b_contours.py:run |
| [ex2_part_c_intersections.png](results/figures/final_report/ex2_part_c_intersections.png) | Copied byte-for-byte | results/figures/ex2_part_c_intersections.png; exercise2/part_c_intersections.py:run |
| [ex2_part_d_gradient_descent.png](results/figures/final_report/ex2_part_d_gradient_descent.png) | Copied byte-for-byte | results/figures/ex2_part_d_gradient_descent.png; exercise2/part_d_gradient_descent.py:run |
| [ex2_part_f_momentum_subplots.png](results/figures/final_report/ex2_part_f_momentum_subplots.png) | Copied byte-for-byte | results/figures/ex2_part_f_momentum_subplots.png; scripts/prepare_report_updates.py:momentum_panels |

The copied originals have PNG resolution metadata of 300 DPI for 1(e), 2(b),
2(c), 2(d), and 600 DPI for the momentum panels. The new 1(c) image is 600 DPI.
Use approximately full text width for the original plot labels and the three
momentum panels. Images and originals were visually reviewed for readable
titles, axes, legends, contour colour bars and unclipped trajectories/markers.

## Numerical verification

- **1(c):** 441 starts on the unchanged 21×21 square grid; all
  441 converge with Armijo and fallback enabled, tol=1e-6, maxit=200.
  The three computed clusters have counts
  205, 232, 4. Markers use computed
  coordinates and the existing gradient/Hessian classification. The contour
  domain is [-2,2]² and its colour bar is the original objective J.

| point_id | classification | x1 | x2 | value | runs_in_cluster |
| --- | --- | --- | --- | --- | --- |
| 1 | minimum | -1.18530257 | 1.18530257 | -1.49914483 | 205 |
| 2 | minimum | -0.659465157 | 0.659465157 | -0.559928824 | 232 |
| 3 | saddle | -0.742325362 | 0.742325365 | -0.555735152 | 4 |

- **1(e):** initial point (-2,1), alpha=1, B0=.01 I; pure fixed-step Newton
  5 updates and BFGS 12.
  All accepted Newton steps are labelled `newton`; no fallback is used. Both
  norm histories, unit steps and the 1e-6 tolerance match the semilog figure.
- **2(b):** phi=F1²+F2² on [-2,2]². The two apparent grid minima are
  (1.38, -0.93), (0.35, 0.76), plotted to two decimal places. These finite-grid locations
  are not labelled as exact roots.
- **2(c):** the unit circle centred at (1,0) and y=cos(2x) give graphical
  estimates (0.3522885, 0.7618848) and
  (1.3773376, -0.9260750), consistent with the displayed
  three-decimal labels. Only the original sampled-curve procedure is used.
- **2(d):** start (0,-1), step .05, ||F||₂<1e-6, maxit=200;
  138 accepted updates. The complete history matches the CSV
  and lies inside the original plotted limits, including start/final markers.
- **2(f):** gamma=.1/.3/.7 gives
  64/36/79 accepted updates.
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
\includegraphics[width=\linewidth]{figures/ex1_part_c_contours.png}
```

The repository's normal `scripts/run_all.py` continues to produce the original
strict-Newton experiment. Use the dedicated final-figure command above for
this report selection. No LaTeX file, PDF, References section or Generative AI
appendix is edited or generated by this script.
