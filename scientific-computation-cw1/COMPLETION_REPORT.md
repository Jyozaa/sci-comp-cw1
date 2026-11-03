# Completion report

## A. Implementation status

Paths are under `src/sc_cw1/`. “Requires review” denotes a documented assumption
or unresolved mathematical claim, not a currently failing test. CODE_MAP.md
lists every shared mathematical helper for each part.

| Part | Main implementation files | Status |
| --- | --- | --- |
| 1(a) | `exercise1/model.py` | Complete and tested (code support; written derivation separate) |
| 1(b) | `exercise1/part_b_newton.py`, `exercise1/model.py`, `common/utilities.py:armijo` | Implemented but requires review: objective-Armijo adaptation and explicit fallback |
| 1(c) | `exercise1/part_c_minima.py`, Newton/model/Armijo | Implemented but requires review: numerical minima complete; true global certification unresolved |
| 1(d) | `exercise1/part_d_bfgs.py`, model gradient and Armijo | Implemented but requires review: curvature safeguard convention |
| 1(e) | `exercise1/part_e_comparison.py`, Newton/BFGS/model | Complete and tested |
| 2(a) | `exercise2/model.py` | Complete and tested (code support; written derivation separate) |
| 2(b) | `exercise2/part_b_contours.py`, model, shared Newton/classification | Complete and tested; finite-grid apparent count, not an exhaustive proof |
| 2(c) | `exercise2/part_c_intersections.py`, model residuals | Complete and tested; graphical estimates separated from supplementary refinement |
| 2(d) | `exercise2/part_d_gradient_descent.py`, model | Implemented but requires review: residual-norm stopping interpretation |
| 2(e) | `exercise2/part_e_momentum.py`, `part_d_gradient_descent.py:fixed_descent,residual_state`, model | Implemented but requires review: zero initial momentum and stopping convention |
| 2(f) | `exercise2/part_f_comparison.py`, momentum/shared loop/model | Implemented but requires review: same stopping/initial-memory conventions |

No numerical sub-question is skipped or blocked. No global-optimality proof is
claimed. The full report is intentionally not generated.

## B. Test and execution results

**87 tests passed; 0 failed; 0 skipped.** All nine numerical module CLIs and the
all-experiment script executed successfully, including from outside the repo.
VALIDATION.md gives the actual commands, environment, derivative/quadratic/step
checks, failure cases, output audit and clean-directory reproduction evidence.

Numerical highlights under the documented conventions:

- 1(c): 441 converged starts, two minima, one saddle. Lowest candidate:
  (-1.1853025702,1.1853025702), J≈-1.4991448256, uncertified globally.
- 1(e): fixed Newton and BFGS converge in 5 and 12 updates respectively.
- 2(d): ordinary descent converges in 138 updates to the lower root,
  final residual norm ≈9.4483e-7.
- 2(f): gamma=.1,.3,.7 converge in 64,36,79 updates to the upper root,
  final residual norms ≈9.4292e-7,8.4801e-7,8.4485e-7.
  Gamma=.7 shows more oscillation; it does not outperform gamma=.3.

## C. Actual generated outputs

All figures are 300 DPI PNGs under `results/figures/`:

1. `ex1_part_c_contours.png`
2. `ex1_part_e_newton_vs_bfgs.png`
3. `ex2_part_b_residual_contours.png`
4. `ex2_part_c_intersections.png`
5. `ex2_part_d_gradient_descent.png`
6. `ex2_part_f_momentum_comparison.png`

All 26 CSVs under `results/tables/`:

1. `ex1_part_b_newton_0.csv`
2. `ex1_part_b_newton_1.csv`
3. `ex1_part_b_newton_2.csv`
4. `ex1_part_b_newton_3.csv`
5. `ex1_part_b_summary.csv`
6. `ex1_part_c_multistart.csv`
7. `ex1_part_c_stationary_points.csv`
8. `ex1_part_d_bfgs_history.csv`
9. `ex1_part_d_summary.csv`
10. `ex1_part_e_newton_history.csv`
11. `ex1_part_e_bfgs_history.csv`
12. `ex1_part_e_summary.csv`
13. `ex2_part_b_grid_candidates.csv`
14. `ex2_part_b_minima.csv`
15. `ex2_part_b_resolution_check.csv`
16. `ex2_part_c_graphical_intersections.csv`
17. `ex2_part_c_refined_intersections.csv`
18. `ex2_part_d_gradient_descent_history.csv`
19. `ex2_part_d_summary.csv`
20. `ex2_part_e_momentum_history.csv`
21. `ex2_part_e_summary.csv`
22. `ex2_part_f_gamma_0.1_history.csv`
23. `ex2_part_f_gamma_0.3_history.csv`
24. `ex2_part_f_gamma_0.7_history.csv`
25. `ex2_part_f_summary.csv`
26. `ex2_local_convergence_diagnostics.csv`

`results/manifest.json` records runtime, parameters and output hashes.
RESULTS.md is generated from actual histories. Required documentation, source
audit, dependency snapshot, validation and reviewer handoff are also supplied.

## D. Uncertainties and limitations

REVIEW_REQUIRED.md documents seven items with exact sources, implementation,
skipped work, impact, resolution options and manual checks:

1. True global optimality of the lowest Exercise 1 candidate is unproved.
2. Exercise 2 stopping uses the lecture-supported Euclidean residual norm,
   which the brief itself does not explicitly specify as the statistic.
3. Initial previous momentum is zero by a stated assumption.
4. Objective-J Armijo adapts residual-based lecture material; optional -g
   fallback is explicit and recorded, and absent from fixed-step 1(e).
5. BFGS rejects unsafe curvature explicitly; the brief gives no such policy.
6. Unprescribed tolerances/grids/refinement settings are explicit; finite
   searches and graphical sampling do not certify completeness.
7. Euclidean squared residual matches the printed factor 2 and lectures;
   the separate report still needs analytical derivations, listings/references
   and its 15-page-format check. Black regions were not reconstructed.

## E. Exact reproduction commands

From the project directory, create the environment and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install . --no-deps
python -m pytest -q
python scripts/run_all.py
```

The existing verified environment can also be used directly:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -W error scripts/run_all.py
```

README.md lists every independent question command and how to choose an
alternative output directory or reinstall after changes.

## F. GitHub review handoff

Send REVIEW_HANDOFF.md with the repository URL. It identifies what is
implemented/verified, the remaining mathematical decisions, numerical limits,
and the exact code sections needing scrutiny: derivative factors, objective
Armijo and fallback branches, expanded inverse-BFGS products/curvature,
unit-step comparison configuration, 2J.T F and momentum-memory update, stopping
boundaries and geometry/clustering completeness. No files were pushed to GitHub.
