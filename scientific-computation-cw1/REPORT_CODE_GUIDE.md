# Guide for the separately written report

This repository supplies code, computations and figures, not the final report.
The original brief limits the single PDF to 15 pages. Show the analytical work
and numerical interpretation in your own report; test output alone is not a
derivation. Original source page numbers below refer to `gpt_copy.pdf`.

## What to include for each question

All code paths in this table are under `src/sc_cw1/`; output paths are under
`results/`. CODE_MAP.md gives the complete shared-function dependencies.

| Part | Essential source/functions | Figure or numbers to use | Listing requirement and interpretation |
| --- | --- | --- | --- |
| 1(a), original p.2 | `exercise1/model.py:objective,gradient`; `hessian` supports subsequent methods | Analytical first-order equations; independent checks in `tests/test_exercise1.py` | No explicit code listing requested here. Derive both gradient entries using the chain/product rules; set them to zero. Source code comments show the implemented formula but do not replace the derivation. |
| 1(b), original pp.2–3 | `exercise1/part_b_newton.py:newton`; `common/utilities.py:armijo`; `exercise1/model.py:gradient,hessian`; `utilities.checked_evaluation` calculates norms | `tables/ex1_part_b_summary.csv` and one `ex1_part_b_newton_*.csv` for evidence | **Source code explicitly required.** Include Newton and Armijo mathematics together; explain x+=alpha*p as the line-search adaptation. Show or describe fallback branches, and identify which steps use them. |
| 1(c), original p.3 | `exercise1/part_c_minima.py:multistart,distinct_stationary_points,classify_stationary_point,run`; imports the Newton method | `figures/ex1_part_c_contours.png`; `tables/ex1_part_c_stationary_points.csv`, `ex1_part_c_multistart.csv` | No extra listing explicitly requested. Give start grid, tol, clustering threshold, stationary-point eigenvalues and objective values. Count minima separately from the saddle. **Resolve R1 before asserting true global optimality.** |
| 1(d), original p.3 | `exercise1/part_d_bfgs.py:bfgs,inverse_bfgs_update`; analytical gradient from model; `utilities.armijo` for normal runs | `tables/ex1_part_d_summary.csv`, `ex1_part_d_bfgs_history.csv` | **Source code explicitly required.** Include the expanded inverse update, B0=omega I, p=-Bg, s and y definitions, curvature check, and status logic. Do not substitute a direct-Hessian update. Explain why no Hessian or solve is needed. |
| 1(e), original p.3 | `exercise1/part_e_comparison.py:compare,run`; fixed branches of Newton and BFGS | `figures/ex1_part_e_newton_vs_bfgs.png`; `tables/ex1_part_e_summary.csv`, both full histories | No extra listing explicitly requested. State x0=(-2,1), fixed alpha=1, BFGS omega=.01. Declare supplementary gradient tolerance/budget. Explain semilog y and initial k=0, and report actual outcomes rather than general speed claims. |
| 2(a), original p.4 | `exercise2/model.py:F,jacobian`; `phi,grad_phi` support later methods | Four analytically derived Jacobian entries | No explicit code listing requested. Explain J is now the Jacobian, not Exercise 1's objective. Differentiate the residuals by hand. |
| 2(b), original p.4 | `exercise2/part_b_contours.py:grid_minimum_candidates,estimate_minima,run`; `model.phi_grid,phi,grad_phi,hessian_phi`; shared Newton/classification | `figures/ex2_part_b_residual_contours.png`; `tables/ex2_part_b_minima.csv`, `ex2_part_b_grid_candidates.csv`, `ex2_part_b_resolution_check.csv` | No listing explicitly required. Define the Euclidean residual norm, count two apparent interior minima, and label derivative refinement as supplementary. Small φ must be checked against both residuals before calling a point a root. Finite resolution is not an exhaustive proof. |
| 2(c), original p.4 | `exercise2/part_c_intersections.py:graphical_estimates,polyline_intersections,refine_intersections,run`; `model.F` checks residuals | `figures/ex2_part_c_intersections.png`; `tables/ex2_part_c_graphical_intersections.csv` and, separately, `ex2_part_c_refined_intersections.csv` | No listing explicitly required. Explain the circle and y=cos(2x) crossing visually. Graphical polyline estimates have finite discretisation error; distinguish separately refined coordinates and residuals. |
| 2(d), original p.4 | `exercise2/part_d_gradient_descent.py:gradient_descent,fixed_descent,residual_state,run`; `model.F,jacobian`; `experiments.run_all` local-rate diagnostic | `figures/ex2_part_d_gradient_descent.png`; `tables/ex2_part_d_summary.csv`, full history, `ex2_local_convergence_diagnostics.csv` | No listing explicitly required here. State exact printed settings and the chosen residual-norm stopping test. Give 138 updates, lower-root destination and final residual. Use local Hessian factors and path evidence to explain slow linear decay, without claiming severe ill-conditioning or zigzagging when neither is observed. |
| 2(e), original p.4 | `exercise2/part_e_momentum.py:momentum_descent`; **`part_d_gradient_descent.py:fixed_descent`** and `residual_state` | Standalone `tables/ex2_part_e_momentum_history.csv`, `ex2_part_e_summary.csv`; 2(f) supplies required comparison figure | **A listing showing modifications is explicitly required.** Show the three `MOMENTUM CHANGE` lines below and enough surrounding loop context to establish where gradients are evaluated and when memory is retained. State d_-1=0 is an assumption. |
| 2(f), original p.5 | `exercise2/part_f_comparison.py:compare,trajectory_statistics,run`; momentum wrapper and shared loop | `figures/ex2_part_f_momentum_comparison.png`; `tables/ex2_part_f_summary.csv` and all three gamma histories | No additional listing explicitly required. State .1,.3,.7 and identical step/start/tol/budget. Compare 64,36,79 updates and objective rises/reversals; momentum runs reach the upper root, unlike ordinary descent. More momentum does not monotonically improve speed. |

## Keep the algorithm listings reviewable

For 1(b), the substantive work is the H p=-g solve, descent test, Armijo
inequality and accepted update. Do not omit `utilities.armijo` merely because
it lives outside the solver module. `checked_evaluation` computes f, g and
||g||; `SolverResult` only records them. Boilerplate CSV/plotting code does not
need to displace mathematical logic in a space-limited listing.

For 1(d), include **both** `bfgs` and `inverse_bfgs_update`. The latter is the
printed formula, not administrative code. The curvature denominator and all
matrix/outer-product orders must remain visible. The inverse-BFGS quadratic
tests and secant test are useful validation to cite briefly, not a replacement
for showing the update itself.

For 2(e), ordinary descent has `direction = g` with `g = 2*J.T@F`. The actual
shared-loop modifications are:

```python
# Before the loop: d_-1 = 0 unless a caller explicitly supplies memory.
previous = np.zeros(2) if initial_direction is None else vector(initial_direction)

# Inside the loop, g has just been evaluated at the CURRENT x_k.
direction = finite(gamma*previous + g)
xn = finite(x - step_size*direction)

# After evaluating and recording x_{k+1}: retain d_k for the next step.
previous = direction.copy()
```

These are the `MOMENTUM CHANGE 1/2/3` markers in `fixed_descent`; the call wrapper
exposes gamma and initial memory. With gamma=0, the same loop implements 2(d).
There is no additional factor on g, adaptive lambda or look-ahead point.

## Definitions and precision to retain

- All trigonometric arguments use radians. First-order and Jacobian formulas
  are analytical. Central finite differences are independent validation.
- A run is converged only when the relevant norm is strictly below tolerance.
  `nit` is the number of updates; x0 is included separately in histories.
- Hessian eigenvalue classification refers to numerical stationary points;
  near-zero eigenvalues would be inconclusive. Four 1(c) starts approach a
  saddle and must not be counted as a third minimum.
- Normal 1(b/c) experiments explicitly use descent fallbacks; 1(e) does not.
  BFGS uses an inverse matrix approximation and no linear solve.
- Use graphical coordinates at roughly 3 decimal places; separately refined
  coordinates can be given to greater precision with their measured residuals.
- The two Exercise 2 apparent minima have residuals near zero; a general local
  minimum of φ need not be a root. The tests include a nonroot stationary case.
- Read all items R1–R7 in REVIEW_REQUIRED.md and the source checks in
  SOURCE_AUDIT.md before choosing final mathematical/report wording.
