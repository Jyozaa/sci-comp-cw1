# Remaining review items and convention audit — Review 1

The original coursework brief remains authoritative. This revision changes no
prescribed parameter, Newton/BFGS/descent update, stopping rule or fallback
policy. The new global analysis is supplementary. No numerical section is
skipped, and no implementation clarification is needed before reproducing the
required experiments. Remaining work concerns independent proof review and
accurate presentation of the explicit conventions below.

## Classification key

1. **Resolved by the original brief.** Follow the printed requirement.
2. **Resolved by supplied lecture materials.** The brief is silent, but the
   supplied teaching materials support the implementation; identify that source
   rather than attributing the detail to the brief.
3. **Reasonable, explicitly stated implementation convention.** Retain and
   disclose it. An unspecified choice is not, by itself, a correctness defect.
4. **Still requiring clarification or mathematical verification.** Do not
   turn numerical evidence into a theorem. The remaining limited item is the
   exhaustive Exercise 2 count if such a stronger claim is desired; its tasks
   ask for apparent minima and graphical solution estimates, which are supplied.

## R1 — Global optimality in 1(c): resolved by supplementary proof

- **Original concern/source:** original brief p.3 asks for the true global
  minimum over R². A finite grid of starts did not establish this.
- **Resolution:** GLOBAL_MINIMUM_ANALYSIS.md verifies the reviewer-proposed
  orthogonal reduction, derives j' and j'', proves all global minimizers lie
  in (-2,0) on x2=-x1, and excludes stationary points outside [-2,2].
  `exercise1/global_analysis.py` uses exact Fraction intervals, justified Taylor
  remainders and outward rounding. Forty cells cover [-2,2] with exactly three
  unique-root cells and no gaps. Strict rational objective bounds select one
  global minimum. There are exactly two minima and one saddle in the plane.
- **Global result:** the unique minimizer is (a_g,-a_g), where a_g is the unique
  j' root in [-1303253958365,-1303253958364]/2^40. Approximate coordinates are
  (-1.1853025702,1.1853025702), J≈-1.4991448256. The decimal point itself is
  not claimed exact. Full root/cover bounds are exported separately.
- **Implemented/skipped:** the full argument and executable proof are supplied;
  no required numerical work or global-certification step remains skipped.
  Existing multistart rows are preserved, including the less tightly refined
  saddle coordinates. The contour legend now refers to the global proof.
- **Impact:** resolves a substantive mathematical claim; does not change any
  original numerical result or algorithm.
- **Manual review:** independently inspect the algebra, Taylor bounds, sign
  logic, coverage and value ordering. This is a rigorous computer-assisted
  argument, not a formally verified theorem-prover development. Its provenance
  and assumptions must be described in the separately written report.

## R2 — Residual stopping and update budget: 2(d/e/f)

| Detail | Class | Source and retained implementation | Impact |
| --- | --- | --- | --- |
| lambda=.05, x0=(0,-1), tol=1e-6, maxit=200 | 1 | Original pp.4–5; unchanged | Changes trajectory/counts if altered |
| Statistic is ||F||₂, not phi or ||grad_phi|| | 2 | Unit 10 printed slides 10/12 (PDF pp.15/17), Unit 11 notebook `norm(F,2)`; matches `residual_state`/`fixed_descent` | Determines convergence and counts; report must name the statistic |
| Strict residual norm < tol | 3 | Brief does not specify equality for Exercise 2; lecture loop `> tol` differs at equality. Strict `<` was chosen consistently and retained | Equality can change a boundary-case count; required runs are unaffected |
| At most 200 completed updates, x0 not counted | 1 | Printed maxit=200; `range(maxit)` and a final convergence check enforce the budget. Do not copy the notebook's `k <= maxk` extra update | Affects iteration counts and correctness of budget handling |
| Test initial/final allowed state for convergence | 3 | Explicit consistent boundary handling; initial point is history row zero | Prevents mislabelling initial or final-update convergence |

**Checked:** normal GD and momentum use the same residual state/stopping loop.
A small residual gradient at a nonroot is never convergence. Additional tests
force a nonzero constant residual through exactly 200 updates and check 201
history rows. Ordinary/required momentum counts remain 138,64,36,79.
No alternative stopping statistic was substituted; no affected run was skipped.
There is no need to ask for a convention change merely because the brief is
silent. If the lecturer later explicitly specifies another statistic, rerun and
explain resulting count changes rather than rewriting historical evidence.

## R3 — Momentum recurrence and initial direction: 2(e/f)

| Detail | Class | Source and retained implementation | Impact |
| --- | --- | --- | --- |
| d_k=gamma*d_(k-1)+2J.T F; x_next=x-lambda*d_k | 1 | Original p.4; `fixed_descent` evaluates g at current x_k; no extra (1-gamma) factor or look-ahead gradient | Mathematical algorithm and trajectory |
| gamma=.1,.3,.7 in independent runs | 1 | Original p.5; `part_f_comparison.compare` constructs fresh states | Trajectory, destination and count |
| d_-1=(0,0) | 3 | Original does not specify memory. `initial_direction=None` selects zero, explicitly documented and overridable | Trajectory, basin and count; state it in report |
| Save d_k for the next update | 1 | Required by the printed recurrence; `previous=direction.copy()` implements it | Mathematical correctness |

**Checked:** all runs start with zero independent memory; first update matches
ordinary descent. The three changed lines are marked in `fixed_descent` and
shown in REPORT_SNIPPETS.md with source/function attribution. No momentum
variant, memory reset or adaptive step was introduced. Nothing is skipped.

## R4 — Objective Armijo and optional Newton fallback: 1(b/c/d/e)

| Detail | Class | Source and retained implementation | Impact |
| --- | --- | --- | --- |
| Newton H p=-g and gradient-norm stopping | 1 | Original pp.2–3; analytical gradient and numpy.linalg.solve | Mathematical correctness |
| Armijo required in 1(b) | 1 | Original p.3 explicitly requires line search | Required algorithm behavior |
| Damping via x_next=x+alpha*p | 3 | Explicit adaptation reconciling line search with the unscaled displayed Newton equation | Algorithm/trajectory; describe adaptation |
| Start at one and halve rejected steps | 2 | Unit 9 printed slide 6/PDF p.6 and Unit 11 examples | Trajectory and robustness |
| Objective J(x+alpha*p)<=J(x)+c1*alpha*g.T*p | 3 | Standard objective adaptation; Unit 9 slide 9/PDF p.12 uses a residual merit sufficient-decrease condition, not literally this J formula | Correct objective decrease on descent directions; report distinction |
| c1=1e-4, shrink=.5, max_backtracks=50, condition limit=1e12 | 3 | Unspecified numeric controls, explicitly recorded | Robustness/possible termination and counts |
| Optional p=-g fallback | 3 | Library default 'none'; 1(b/c) opt in for singular/ill-conditioned or non-descent cases. Separate step_kind records/reason counts | Basins/counts; not an ordinary Newton step |
| 1(e) x0=(-2,1), fixed alpha=1, omega=.01 | 1 | Original p.3. `step_mode='fixed', fallback='none', curvature_action='fail'` | Required comparison; no hidden line search or fallback |

**Checked:** Armijo does not accept non-descent directions or exhausted searches.
The fixed-step comparison bypasses Armijo entirely (a new monkeypatch test
would fail if it called it, even if the search returned one). Ordinary Newton
can approach a saddle/maximum; a small gradient is not minimum classification.
Fallback rows cannot be described as Newton steps. The global proof concerns
this objective's minimizer, not a theorem of global convergence of the solver.
No numerical functionality was skipped.

## R5 — Inverse BFGS and curvature safety: 1(d/e)

| Detail | Class | Source and retained implementation | Impact |
| --- | --- | --- | --- |
| B0=omega I, p=-Bg, printed expanded inverse update | 1 | Original p.3, cross-checked rendered formula; unchanged | Mathematical correctness |
| Stop strictly at ||g||₂<tol | 1 | Original 1(d) explicitly prints `< tol` | Convergence/counts |
| Require s.T y>0 and >1e-12*||s||*||y|| | 3 | Brief supplies denominator but no safeguard; `inverse_bfgs_update` fails explicitly before division | Safety may terminate a difficult run; no change to accepted update |
| curvature_action='fail' in every delivered BFGS run | 3 | Explicit convention, recorded in parameters; optional 'skip' is unused | Failure status rather than silently altered method |
| No B update after a converged new gradient | 3 | No subsequent direction will use B; iteration stops | No change to iterates/counts |

**Checked:** normal iteration calls no Hessian and no linear-system solve.
There are no resets, damped curvature vectors or hidden denominator edits.
Armijo alone is not claimed to guarantee positive curvature. All delivered
BFGS updates are ordinary inverse updates until the final converged point,
not skipped/repaired updates. The printed formula is unchanged. Nothing is
skipped in the required runs.

## R6 — Unprescribed tolerances, sampling and completeness

| Detail | Class | Retained implementation | Impact |
| --- | --- | --- | --- |
| Exercise 1 tol=1e-6, maxit=200; demo omega=.01 | 3 | Supplementary settings, distinct from 1(e)'s prescribed omega=.01 | Counts/termination; state explicitly |
| 1(c) starts in [-2,2]² | 1 | Original p.3 | Required experiment domain, not an a priori bound on global minimizers |
| 21×21 starts, distance threshold 1e-4, eigenvalue threshold 1e-7 | 3 | Original settings preserved; global proof is separate | Numerical clustering/classification and presentation |
| 401×401 residual grid, 201/401/801 resolution checks | 3 | Apparent interior minima; separate gradient-tolerance-1e-7 refinement | Sampling accuracy; finite-grid count alone is not a theorem |
| 2001 samples on each zero curve; separate residual-1e-12 bisection | 3 | Plotted-polyline estimates retained separately from refinement | Presentation precision; do not label refined accuracy as graphical |
| Exhaustive Exercise 2 root/minimum count | 4 if asserted | Not proved by these scans; brief asks for apparent minima/graphical estimates, which are supplied | Would affect a stronger mathematical claim; no such claim is made |

No prescribed setting was replaced. Root isolation in the supplementary
Exercise 1 proof uses exact rational signs and analytic tails, not a denser
search. Only Exercise 2's stronger optional completeness claim remains
mathematically unproved; no clarification is needed for the requested visual
interpretation. Verify that the final report preserves that distinction.

## R7 — Norm interpretation and separate report

The original p.4 writes ||F||² without a norm subscript. **Class 2:** Unit 10
explicitly defines the sum of squares and derives 2J.T F, also matching the
printed momentum recurrence. Euclidean interpretation is supported, not an
extra square. Reuse of J across exercises is original notation (**class 1**),
not a typo. It needs a clear local definition in the report, not a code change.

Original p.1's 15-page limit and source listings in 1(b),1(d),2(e) are **class 1**.
REPORT_SNIPPETS.md supplies literal excerpts and identifies omitted setup,
helpers, guards and recording. The actual report, personal derivations,
references, appropriate proof explanation and page-count/layout verification
remain to be prepared. This is presentation/submission work, not an unresolved
algorithmic defect. No full report was generated, as requested. Source black
regions remain ignored; source hashes were rechecked against SOURCE_AUDIT.md.
