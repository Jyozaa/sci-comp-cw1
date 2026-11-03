# Code map

Paths below are relative to `src/sc_cw1/`, unless otherwise stated. A `run`
function generates a question's outputs; its solver/model functions contain the
mathematical work. All plotting/export helpers are listed at the end.

## Exercise 1

| Part | Implementation, functions and responsibilities | Outputs under results/ |
| --- | --- | --- |
| 1(a) | `exercise1/model.py`: `objective` implements J; `gradient` contains the hand-derived components and chain-rule explanation; `hessian` gives the exact second derivatives; `numerical_hessian` centrally differences `gradient`; `objective_grid` evaluates the identical J equation for plots. `common/utilities.py:finite_difference_jacobian` supplies central differences. | Derivative verification in `tests/test_exercise1.py`; no prescribed figure |
| 1(b) | `exercise1/part_b_newton.py:newton` solves H p=-g with `numpy.linalg.solve`, checks conditioning/descent, selects Armijo or fixed steps, records fallback reasons and convergence; `run` demonstrates four starts. `exercise1/model.py:objective,gradient,hessian` supplies J/g/H. **`common/utilities.py:armijo` implements the entire sufficient-decrease calculation and step-halving loop.** `checked_evaluation` computes f,g,||g|| safely. | `tables/ex1_part_b_newton_0.csv` through `_3.csv`; `ex1_part_b_summary.csv` |
| 1(c) | `exercise1/part_c_minima.py:multistart` constructs the full square grid and calls `part_b_newton.newton` with explicit fallback; `distinct_stationary_points` excludes failed/nonstationary runs and clusters by distance; `classify_stationary_point` tests ||g|| and Hessian eigenvalues; `run` selects the lowest minimum found, exports per-start outcomes and draws contours. It imports `exercise1/model.py:objective,gradient,hessian,objective_grid` and all 1(b) mathematics, including `utilities.armijo`. | `figures/ex1_part_c_contours.png`; `tables/ex1_part_c_multistart.csv`, `ex1_part_c_stationary_points.csv` |
| 1(d) | `exercise1/part_d_bfgs.py:bfgs` initialises B=omega I, takes p=-Bg, computes s and y, records curvature and exits; **`inverse_bfgs_update` implements every term of the printed inverse update**; `run` executes the standalone demonstration. `exercise1/model.py:objective,gradient` supplies f/g, with no Hessian import. `utilities.armijo` selects normal steps; `checked_evaluation` supplies scalar diagnostics. | `tables/ex1_part_d_bfgs_history.csv`, `ex1_part_d_summary.csv` |
| 1(e) | `exercise1/part_e_comparison.py:compare` fixes start (-2,1), alpha=1 and BFGS omega=.01; explicitly disables Newton fallback and BFGS update skipping; `run` plots every gradient norm including k=0. Calls `part_b_newton.newton`, `part_d_bfgs.bfgs,inverse_bfgs_update`, and `model.objective,gradient,hessian`; their fixed branches bypass `utilities.armijo`. | `figures/ex1_part_e_newton_vs_bfgs.png`; `tables/ex1_part_e_newton_history.csv`, `ex1_part_e_bfgs_history.csv`, `ex1_part_e_summary.csv` |

## Exercise 2

| Part | Implementation, functions and responsibilities | Outputs under results/ |
| --- | --- | --- |
| 2(a) | `exercise2/model.py:F` returns the two residuals; `jacobian` contains all four analytical entries; `phi` computes F.T F; `grad_phi` computes 2J.T F. `hessian_phi` is supplementary for minimum classification/local-rate diagnostics, not descent updates. `residual_grid,phi_grid` evaluate the same equations on plotting arrays. | Derivative verification in `tests/test_exercise2.py`; no prescribed figure |
| 2(b) | `exercise2/part_b_contours.py:grid_minimum_candidates` scans eight neighbours of interior grid pixels; `estimate_minima` refines candidates and tests residuals separately from stationarity; `run` draws isocontours on the full prescribed square and labels estimates. It uses `model.F,phi,grad_phi,hessian_phi,phi_grid`, `exercise1/part_b_newton.py:newton`, `exercise1/part_c_minima.py:distinct_stationary_points,classify_stationary_point`, and `utilities.armijo,checked_evaluation`. This Newton refinement is supplementary and does not change required descent trajectories. | `figures/ex2_part_b_residual_contours.png`; `tables/ex2_part_b_grid_candidates.csv`, `ex2_part_b_minima.csv` |
| 2(c) | `exercise2/part_c_intersections.py:graphical_estimates` constructs the circle and cosine zero curves; `polyline_intersections,cross2` compute intersections of the **plotted segments**; `refine_intersections` separately bisects the scalar equation after eliminating y; `run` labels graphical coordinates and exports both kinds of estimates. `exercise2/model.py:F` checks the two residuals. | `figures/ex2_part_c_intersections.png`; `tables/ex2_part_c_graphical_intersections.csv`, `ex2_part_c_refined_intersections.csv` |
| 2(d) | `exercise2/part_d_gradient_descent.py:gradient_descent` selects gamma=0 and the exact settings; **`fixed_descent` contains the update loop**, fixed lambda, residual-norm stopping and history; **`residual_state` calculates 2J.T F, phi and both norms** from `model.F,jacobian`; `add_residual_background` draws `model.phi_grid`; `run` saves the trajectory. `utilities.plot_path` marks initial/intermediate/final iterates. | `figures/ex2_part_d_gradient_descent.png`; `tables/ex2_part_d_gradient_descent_history.csv`, `ex2_part_d_summary.csv` |
| 2(e) | `exercise2/part_e_momentum.py:momentum_descent` validates gamma, exposes the initial direction and delegates to **`part_d_gradient_descent.py:fixed_descent`**, where `MOMENTUM CHANGE 1/2/3` identify initial memory, `direction=gamma*previous+g`, and memory retention; `residual_state` supplies the analytical 2J.T F via `model.F,jacobian`. `run` executes gamma=.3 as a standalone illustration. Read **both files** for the requested listing. | `tables/ex2_part_e_momentum_history.csv`, `ex2_part_e_summary.csv` |
| 2(f) | `exercise2/part_f_comparison.py:compare` runs fresh independent solver states for .1,.3,.7 with prescribed settings; `trajectory_statistics` measures path length, objective rises and step reversals; `run` draws full trajectories plus a solution detail. Calls `part_e_momentum.momentum_descent`, `part_d_gradient_descent.fixed_descent,residual_state,add_residual_background`, `model.F,jacobian,phi_grid`, and `utilities.plot_path`. | `figures/ex2_part_f_momentum_comparison.png`; `tables/ex2_part_f_gamma_0.1_history.csv`, `_0.3_history.csv`, `_0.7_history.csv`, `ex2_part_f_summary.csv` |

## Shared recording and orchestration

- `common/utilities.py:SolverResult` stores copied coordinates, gradient norms,
  step lengths, parameters and exit statuses. `vector,finite,validate_controls`
  validate shape/values/controls. These contain no hidden optimisation update.
- `write_csv,save_history,summary` export real computed values and statuses;
  `output_dirs` manages destinations; `pyplot,save_figure,plot_path` manage
  headless 300 DPI plotting and trajectory markers; `module_cli` supplies each
  question's `--output` command interface.
- `experiments.py:run_all` calls all nine numerical `run` functions; analytical
  1(a)/2(a) are exercised through their imports and tests. It adds resolution
  checks using `part_b_contours.estimate_minima`, and local convergence diagnostics
  using `model.hessian_phi` and `numpy.linalg.eigvalsh`. It writes
  `tables/ex2_part_b_resolution_check.csv`, `tables/ex2_local_convergence_diagnostics.csv`,
  `manifest.json` and `RESULTS.md`. `markdown_table` formats the computed digest.
- `scripts/run_all.py:main` is the single entry point, including source-tree
  import setup. It contains no numerical algorithm.

## Verification locations

- `tests/test_exercise1.py`: direct values, objective/gradient/Hessian finite
  differences, symmetric Hessian, grid/scalar agreement, converged gradient
  norms, accepted Armijo inequalities, prescribed fixed steps, classification
  and deduplication.
- `tests/test_numerical_methods.py`: exact Newton quadratic step, independently
  known BFGS minima, expanded-versus-compact inverse update and secant equation,
  positive definiteness, no BFGS linear solve, curvature/conditioning/descent
  safeguards, exhausted searches, iteration limits and floating-point failures.
- `tests/test_exercise2.py`: independent Jacobian/gradient/Hessian differences,
  hand-computable ordinary/momentum steps, zero/custom initial memory, exact
  settings and budgets, failure handling, graphical estimates and classification
  of apparent minima separately from residual zero.
