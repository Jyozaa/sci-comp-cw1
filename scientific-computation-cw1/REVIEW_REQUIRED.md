# Manual checks and numerical conventions

No required numerical algorithm is skipped. The remaining limitations concern
what the numerical evidence establishes and how it is described in the report.
The original brief is authoritative; assumptions are not additional instructions.

| Part | Convention or limitation | Source/support and report action |
| --- | --- | --- |
| 1(c) | Lowest minimum found near (-1.185303,1.185303), J≈-1.499145, is a numerical global candidate. | The brief asks for the true global minimum. Multiple-start objective comparisons support the numerical answer, but finite searches do not prove global optimality over R² or completeness. Use that precise wording; review whether the numerical justification meets the lecturer's expected standard. No formal proof is provided in this course-scope version. |
| 1(b/c) | Optional -gradient fallback is enabled in the multiple-start experiment. | Gradient descent is taught in Unit 10; the brief does not prescribe this safeguard. `step_kinds` and the multistart CSV count it separately. Identify these as fallback steps, not ordinary Newton steps. Library default is off; 1(e) disables it. |
| 1(b/d) | Objective Armijo uses c1=1e-4, alpha initially 1, halving, 50 backtracks. | Unit 9/11 teach step-halving for a residual merit function. Using J(x+alpha*d)<=J(x)+c1*alpha*grad.T*d is the stated objective adaptation. The brief requires Armijo but does not prescribe these constants. Non-descent/exhausted searches fail explicitly. |
| 1(d) | Require s.T y>0 and >1e-12*||s||||y|| before the printed inverse update. | The brief divides by s.T y without specifying a safeguard. Failure is reported; there is no skip/reset/damping/denominator edit. All required comparison steps remain unit length. |
| Exercise 1 | tol=1e-6, maxit=200; 21×21 multistart grid; distance threshold 1e-4. | Brief leaves these numbers unspecified, except the start square. State them as implementation choices. The basic 2×2 Hessian test uses a 1e-7 threshold; nearly degenerate cases would remain inconclusive. |
| 2(b/c) | Apparent minima and crossings are finite-grid/graphical estimates. | These tasks ask for visual analysis. The residual grid has spacing .01; each zero curve has 2001 samples. No refined root precision or exhaustive count is claimed. A grid minimum with nonzero residual must not be labelled an exact root. |
| 2(d/e/f) | Stop strictly at ||F||₂<1e-6; at most 200 updates; x0 is not an update. | Unit 10 slides 10/12 and Unit 11's notebook support residual-norm stopping; the brief names tol but not its statistic. Strict equality handling is a retained convention. Do not copy the notebook's potential extra update from `k <= maxk`. Initial/final-allowed states are checked. |
| 2(e/f) | Initial previous direction is zero and every gamma run has fresh memory. | Not specified by the brief; explicitly stated convention. The exact printed recurrence is retained, with no alternative momentum definition or adaptive step. |
| 2(a/b/e) | phi is the Euclidean squared residual, F1²+F2². | Unit 10 explicitly defines the sum of squares and derives 2J.T F; this also matches the printed momentum term. The original notation has no norm subscript. |

The numerical conventions affect iteration counts/basins, so they must appear
in the report; they were not changed to force the regression results. Geometric
sampling affects displayed coordinates and residuals, not the required solver
trajectories. A singular Hessian, unsafe BFGS update, nonfinite iterate or
exhausted line search is a failure status, never a minimum.

The written submission remains separate: supply the analytical derivations for
1(a)/2(a), source listings for 1(b)/1(d)/2(e), interpretation of each required
figure, and appropriate references. Check the original single-PDF, at-most-15-page
requirement. The repository does not create that report. Source PDFs/notebooks
are kept outside the repository; no obscured content has been reconstructed.
