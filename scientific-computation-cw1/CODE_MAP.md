# Code map

Paths are relative to `src/sc_cw1/`. All algorithm functions return plain
dictionaries of arrays. Basic CSV/plot helpers contain no numerical update.

| Part | Functions/files and every mathematical dependency | Required output / report use |
| --- | --- | --- |
| 1(a) | `exercise1/model.py:objective,gradient` implement the original J and analytical partial derivatives; `hessian` gives second derivatives; `numerical_hessian` checks the gradient using central differences. `objective_grid` evaluates the identical formula on arrays. | Written first-order derivation; derivative tests. No figure/listing explicitly requested here. |
| 1(b) | **`exercise1/part_b_newton.py:newton,armijo`** contain the entire Newton solve and objective step-halving condition. They use `model.objective,gradient,hessian` (or the permitted numerical Hessian), `numpy.linalg.solve` and explicit status checks. There is no numerical helper in another directory. | Source listing explicitly required; standalone `run` prints an example. The different-start evidence comes from 1(c). |
| 1(c) | `exercise1/part_c_minima.py:multistart` creates the square grid and calls `part_b_newton.newton` with recorded optional -gradient fallback. `classify_stationary_point` uses `model.gradient,hessian` and the simple 2×2 second-derivative test. `distinct_stationary_points` merges converged points by distance and compares `model.objective`. `run` draws `model.objective_grid`. | `figures/ex1_part_c_contours.png`; `tables/ex1_part_c_multistart.csv`, `ex1_part_c_stationary_points.csv`. Lowest candidate is numerical evidence, not a global proof. |
| 1(d) | **`exercise1/part_d_bfgs.py:bfgs,inverse_bfgs_update`** initialise B, calculate delta/s/y and apply the printed expanded inverse update. They use `model.objective,gradient` and `part_b_newton.armijo` only when line search is enabled. No Hessian evaluation/linear solve occurs in BFGS. | Source listing explicitly required; standalone `run` prints an example. Required comparison histories are in 1(e). |
| 1(e) | `exercise1/part_e_comparison.py:compare` fixes both starts/unit steps and BFGS omega=.01; calls `part_b_newton.newton` and `part_d_bfgs.bfgs` with line search off and Newton fallback off. Those use the 1(a) model; `run` plots gradient norms including k=0. | `figures/ex1_part_e_newton_vs_bfgs.png`; `tables/ex1_part_e_newton_history.csv`, `ex1_part_e_bfgs_history.csv`; summary below. |
| 2(a) | `exercise2/model.py:F,jacobian` implement the two equations/four analytical derivatives; `phi,grad_phi` compute F1²+F2² and 2J.T F. `residual_grid,phi_grid` evaluate the same equations for plots. | Written Jacobian derivation and basic derivative tests. No figure/listing explicitly requested here. |
| 2(b) | `exercise2/part_b_contours.py:grid_minimum_candidates` compares each interior pixel with its eight neighbours using `model.phi_grid`; `run` plots the full square and uses `model.phi,F` to report approximate values/residuals. There is no optimiser/refinement here. | `figures/ex2_part_b_residual_contours.png`; `tables/ex2_part_b_minima.csv`. Count apparent minima; do not claim grid points are exact roots. |
| 2(c) | `exercise2/part_c_intersections.py:graphical_estimates` samples the circle and cosine zero curves, finds sampled circle-residual sign changes along the cosine and makes one linear interpolation per crossing. `run` plots both curves and checks approximate residuals with `model.F`. | `figures/ex2_part_c_intersections.png`; `tables/ex2_part_c_graphical_intersections.csv`. Graphical precision only, with no iterative root refinement. |
| 2(d) | **`exercise2/part_d_gradient_descent.py:gradient_descent`** contains residual/Jacobian evaluation, **2J.T F**, the fixed update, stopping and history lists in one function. Defaults gamma=0. It uses only `model.F,jacobian`; `residual_background` uses `model.phi_grid` for context; `run` plots all iterates. | `figures/ex2_part_d_gradient_descent.png`; `tables/ex2_part_d_gradient_descent_history.csv`; summary below. |
| 2(e) | `exercise2/part_e_momentum.py:momentum_descent` selects 0<gamma<1 and calls the **same `part_d_gradient_descent.gradient_descent` loop**. The marked lines initialise `previous=0`, compute `d=gamma*previous+g`, then retain `previous=d.copy()`. The Jacobian/gradient calculation stays visible in that loop. | Source listing of the modifications explicitly required. No extra demonstration table; 2(f) gives the required gamma experiments. |
| 2(f) | `exercise2/part_f_comparison.py:compare` starts fresh `.1,.3,.7` runs through `part_e_momentum.momentum_descent` and the shared 2(d) loop/model. `run` uses `part_d_gradient_descent.residual_background` and directly plots all paths/start/end markers. | `figures/ex2_part_f_momentum_comparison.png`; `tables/ex2_part_f_gamma_0.1_history.csv`, `_0.3_history.csv`, `_0.7_history.csv`; summary below. |

`experiments.py:run_all` executes exactly the six figure-producing experiments,
collects the required run statuses/counts/final values in
`tables/comparison_summary.csv`, and regenerates RESULTS.md. `table` only formats
Markdown. The ordinary NumPy count of objective increases helps describe
momentum oscillation; no extra experiment is run.

`common/utilities.py` contains `output_dirs,write_csv,save_history,summary,
save_figure,module_cli` and a noninteractive Matplotlib backend. Histories are
exported with iteration zero; CSV schema changes are explained in the audit.
`scripts/run_all.py` supplies source-tree imports and the output-directory option.
There are exactly six figures and eleven CSVs.

The three test modules check the models/derivatives, independent quadratic and
hand-computable updates, Armijo, inverse-BFGS secant identity, failures, prescribed
settings, update budgets, clustering, and graphical-estimate precision.
REPORT_SNIPPETS.md contains literal excerpts of the current functions; the
separate report still needs the analytical derivations and numerical discussion.
