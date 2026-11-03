# Review required

No numerical sub-question was blocked or skipped. All prescribed experiments
execute. **The true-global-minimum certification asked for in 1(c) remains
unresolved.** The other items below are explicit conventions, not claims that
the original brief specified them. Test success does not settle these choices.

## R1 — Global optimality and search completeness: 1(c)

**TODO R1:** supply an independently checked global-optimality argument before
upgrading the numerical candidate to a certified true global minimum.

- **Source of uncertainty:** original brief p.3 asks which minimum is the true
  global minimum over R². Starts are restricted to [-2,2]², but that does not
  restrict the optimisation problem. A finite multistart grid cannot certify
  completeness, and no supplied theorem certifies this particular objective.
- **Implemented:** 441 Newton/Armijo searches, stationary-point clustering,
  Hessian classification, two distinct local minima and a saddle. Lowest
  candidate approximately (-1.1853025702,1.1853025702), J≈-1.4991448256.
  Figures and tables label this as the lowest minimum **found**, uncertified.
- **Skipped/unresolved:** no rigorous global-optimality proof, interval enclosure
  of all stationary points, or exhaustive enumeration over the plane.
- **Impact:** substantive mathematical completeness of the global claim;
  not a known bug in the objective or local solvers.
- **Resolution options:** prove a bounded sublevel set contains every better
  candidate, then analytically/rigorously enumerate candidates; or obtain a
  lecturer-approved numerical-evidence standard. Denser/larger searches can
  strengthen evidence but do not become a proof by themselves.
- **Manual verification:** independently check derivatives, stationary-point
  classification, and a valid argument before calling the candidate global.

## R2 — Stopping statistic and equality: 2(d), 2(e), 2(f)

- **Source of uncertainty:** original pp.4–5 prescribe tol=1e-6 but do not name
  the statistic. Unit 10 printed slides 10/12 (PDF pp.15/17) and the Unit 11
  notebook use `norm(F,2)` against tol; notebook loop boundaries also differ
  from an exact 200-update budget.
- **Implemented:** strict `||F||_2 < 1e-6`, with at most **200 updates**, checking
  convergence at the initial point and after the final permitted update.
  Initial x0 is history row zero. The residual gradient is never substituted.
- **Skipped:** alternative stopping statistics and the notebook's potential
  201st update. No required run is skipped.
- **Impact:** changes iteration counts and possibly reported convergence;
  affects interpretation/correctness if a different statistic was intended.
- **Resolution options:** retain this lecture-supported convention and state it,
  or obtain clarification and rerun all affected experiments consistently.
- **Manual verification:** confirm the intended norm/statistic, strict versus
  non-strict boundary, and counting convention in the report. Current counts
  138,64,36,79 follow the documented rule, not an unspecified generic tol.

## R3 — Initial momentum: 2(e), 2(f)

- **Source of uncertainty:** original p.4 recurrence references d_(k-1) without
  specifying the initial memory. The corrected reference also flags this.
- **Implemented:** d_-1=(0,0), exposed through `initial_direction`; fresh memory
  for every gamma. First step equals ordinary gradient descent. The recurrence
  has **no (1-gamma) gradient scaling** and no Nesterov/look-ahead evaluation.
- **Skipped:** no required functionality; other initial-memory experiments have
  not replaced the required runs.
- **Impact:** algorithmic convention affects trajectory, basin and counts.
- **Resolution options:** state zero initial momentum explicitly or rerun after
  obtaining an intended nonzero initial direction.
- **Manual verification:** inspect the three `MOMENTUM CHANGE` markers in
  `fixed_descent` and the wrapper; make the assumption visible in the listing.

## R4 — Objective Armijo and descent safeguard: 1(b), 1(c), 1(d)

- **Source of uncertainty:** original equation (2), p.2, is unscaled; p.3 also
  requires Armijo. Unit 9 printed slide 6 (PDF p.6) halves steps based on a
  system residual; slide 9 (PDF p.12) gives residual-merit sufficient decrease.
  Neither is literally an objective-J Armijo formula for this optimisation
  problem. The brief does not prescribe handling an indefinite Hessian.
- **Implemented:** objective Armijo
  J(x+alpha*p) <= J(x)+c1*alpha*g.T*p, c1=1e-4, shrink=.5,
  first alpha=1, up to 50 halvings. Non-descent directions are rejected.
  Library Newton defaults to no fallback; **1(b/c) runs explicitly enable -g
  fallback** for singular/condition-limit or non-descent cases and record it.
  Condition limit is 1e12. Regular Newton and fallback counts are separate.
- **Skipped:** no required numerical feature. There is no automatic matrix
  shift, hidden damping or forced acceptance. **1(e) uses none of these
  adaptations**, exactly fixed alpha=1 and fallback='none'.
- **Impact:** the Armijo formulation and optional fallback change robustness
  and basins; treating all recorded steps as ordinary Newton would be inaccurate.
- **Resolution options:** justify the standard objective sufficient-decrease
  adaptation and explicitly describe the fallback, or run strict Newton and
  report its failures. No guarantee of global convergence is claimed.
- **Manual verification:** check accepted inequalities, the distinction between
  Unit 9's residual merit and objective J, and the fallback disclosure in report
  text/listings. A formal external reference may be needed for the adaptation.

## R5 — BFGS curvature safety: 1(d), 1(e)

- **Source of uncertainty:** original p.3 divides by s.T y but gives no safeguard.
  Armijo alone does not guarantee positive BFGS curvature for every objective.
- **Implemented:** the printed expanded **inverse-Hessian** update; require
  s.T y > 0 and >1e-12*||s||*||y|| before division. Failure is explicit.
  An opt-in `curvature_action='skip'` exists for supplementary work but all
  delivered normal/required BFGS runs use `'fail'`; none skips an update or
  resets B. Once the new gradient meets tolerance no further B update is needed.
- **Skipped:** no required feature or run. Invalid curvature is not repaired
  using a fabricated y, hidden damping, Hessian or linear solve.
- **Impact:** supplementary numerical safety may end a difficult run earlier
  than the algebraically undefined/unstable printed iteration.
- **Resolution options:** retain and document fail-fast behavior; investigate
  stronger line-search conditions only in separately labelled experiments.
- **Manual verification:** expanded formula order, denominators and secant
  equation; confirm 1(e) histories show unit steps and no skipped updates.

## R6 — Supplementary numerical settings: Exercise 1 and geometry estimates

- **Source of uncertainty:** original p.3 does not give Exercise 1 tolerance,
  budget, grid density, cluster/eigenvalue thresholds, or 1(d) demo omega;
  pp.4–5 do not prescribe plot sampling or intersection-refinement algorithms.
- **Implemented:** Exercise 1 tol=1e-6, maxit=200; 1(c) grid 21×21, clustering
  radius 1e-4, Hessian eigenvalue threshold 1e-7. 1(d) demonstration omega=.01.
  Residual plot uses 401×401 points; refinement uses gradient tolerance 1e-7;
  201/401/801 resolution checks find the same two apparent minima. Circle and
  cosine are each drawn from 2001 points; separate scalar bisection refines
  intersections to residual <1e-12. No values replace printed parameters.
- **Skipped/unresolved:** finite sampling cannot exclude additional narrow or
  tangent features. Interior pixel minima exclude constrained boundary minima;
  the task is interpreted as unconstrained local minima visible in the region.
- **Impact:** estimates/presentation and completeness limits; solver stopping
  choices affect reported counts. No exhaustive global/local count is certified.
- **Resolution options:** increase resolution and test tolerance sensitivity as
  supplementary studies; use analytical/validated methods for completeness.
- **Manual verification:** assess whether these resolutions/tolerances support
  the report's precision, and do not present refined roots as graphical accuracy.

## R7 — Norm interpretation and written work: 1(a), 2(a/b/e), final report

- **Source of uncertainty:** original p.4 prints ||F||² without a norm subscript.
  J is an objective in Exercise 1 and a Jacobian in Exercise 2; this is notation
  reuse, not a typo. Analytical derivations/listings must appear in the separately
  written submission, which is limited to 15 pages (original p.1).
- **Implemented:** Euclidean φ=F1²+F2² and ∇φ=2J.T F, matching Unit 10 and the
  printed momentum formula. Hand-derived formulas/docstrings, source map and
  report/listing guide are supplied. `hessian_phi` is supplementary diagnostics.
- **Skipped:** no automatic full coursework report or written submission, as
  requested. Blacked-out original regions were ignored, not reconstructed.
- **Impact:** Euclidean convention is strongly supported; remaining work is
  report presentation/justification, plus the substantive R1 global claim.
- **Resolution options/manual verification:** explicitly define the norm and
  each exercise's J, supply personal derivations, include the requested Newton,
  inverse-BFGS and momentum-modification listings, and check report precision,
  references and page count against the original brief.
