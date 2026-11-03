# Independent review handoff

The repository implements all code-related COMP5930M CW1 parts: hand-derived
objective derivatives; Newton/Armijo with recorded optional descent fallback;
multistart classification; the printed expanded inverse-BFGS update; prescribed
unit-step comparison; nonlinear residual/Jacobian; residual contours and
graphical intersections; fixed-step gradient descent and printed momentum.
It includes tests, six 300 DPI figures, 26 computed CSVs and a runtime manifest.
CODE_MAP.md locates every function/output; REPORT_CODE_GUIDE.md locates the
three explicitly requested source listings. No complete report is supplied.

## Verified evidence

Read VALIDATION.md for the actual final test run and environment. Independent
central differences check both exercises' derivatives, including direct
objective second differences. Quadratic tests check Newton/BFGS; BFGS secant
and expanded-versus-compact updates agree. Hand-computable steps check the
residual-gradient factor of two and momentum recurrence. Tests cover Armijo,
fixed steps, budgets, statuses, curvature and numerical failures.

All coursework experiments were actually run. The 441-start Newton search
found two local minima and one saddle; fixed Newton/BFGS converged in 5/12
steps. Ordinary descent converged in 138 updates to the lower root. Momentum
.1/.3/.7 converged in 64/36/79 updates to the upper root. Read the actual CSV
residuals and histories before making broader convergence claims.

## Mathematical decisions needing review

1. **Global optimality is unresolved.** The lowest minimum found has
   J≈-1.4991448256 at (-1.1853025702,1.1853025702). No rigorous proof that it
   is the true global minimum or that the finite search is exhaustive is supplied.
2. Exercise 2 uses ||F||_2<1e-6, supported by Unit 10 but not explicitly
   specified as the stopping statistic by the brief. Initial momentum is zero
   by convention. Both choices affect trajectories/counts and are disclosed.
3. Objective-J Armijo is a standard adaptation of the residual-based lecture
   line search; ordinary Newton and optional -g steps are distinguished. The
   inverse-BFGS curvature failure policy is an additional safety convention.
4. Numerical settings not supplied for Exercise 1/geometry are documented;
   finite grid and graphical estimates are not certified exhaustive counts.

REVIEW_REQUIRED.md gives exact sources, implemented/skipped work, impact,
resolution options and manual checks for each item. The report still needs
personal analytical derivations, the requested code listings, references and
the original 15-page-format check.

## Code and numerical review priorities

- `exercise1/model.py`: sine-square chain rule, mixed Hessian terms and factors.
- `common/utilities.py:armijo` and `exercise1/part_b_newton.py:newton`: actual
  objective decrease, non-descent handling, condition/fallback records, finite
  failures and final-update stopping. No acceptance relaxation is used.
- `exercise1/part_d_bfgs.py:inverse_bfgs_update,bfgs`: matrix multiplication order,
  inverse formulation, tiny/nonpositive curvature and no hidden solve/reset.
- `exercise1/part_e_comparison.py:compare`: prescribed fixed alpha=1 and
  omega=.01, without applying the normal line-search/fallback branch.
- `exercise2/part_d_gradient_descent.py:residual_state,fixed_descent` and
  `part_e_momentum.py:momentum_descent`: 2J.T F, gradient evaluated at x_k,
  d_-1=0, exact memory recurrence, fresh independent runs, residual stopping,
  and no extra 201st update. The momentum wrapper alone is not the full listing.
- Geometry scanning/clustering/refinement: only converged stationary points
  are classified; graphical polyline locations are separate from bisection.
  Assess tolerance/resolution sensitivity without confusing it with proof.

Reproduce with `python -m pytest -q` and `python scripts/run_all.py` after the
README installation. No GitHub commit, push or coursework submission was made.
