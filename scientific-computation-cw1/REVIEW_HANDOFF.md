# Independent review handoff — after Review 1

Baseline: f21121f. The previous independent review found no confirmed core
algorithmic errors and reproduced the main results. This focused revision
preserves all objective/derivative, Newton, inverse-BFGS, fixed-step,
residual/Jacobian, ordinary-descent and printed-momentum implementations.
All prescribed parameters and original 87 tests are unchanged. Changes are
supplementary mathematical analysis, proof data, report excerpts and reporting.
No changes in this revision were committed or pushed.

## What is now resolved mathematically

GLOBAL_MINIMUM_ANALYSIS.md independently verifies the proposed orthogonal
reduction and j'/j'', proves all global minimizers lie on x2=-x1 with a in
(-2,0), and excludes all reduced stationary points outside [-2,2].

`exercise1/global_analysis.py` supplies an exact Fraction interval proof:
positive-series exponential tails, degree-95 sine/cosine Taylor remainders,
Lipschitz extension and outward dyadic rounding enclose j'/j'' on rational
cells. Forty cells cover [-2,2] without gaps. Every cell excludes a root or
proves exactly one by strict endpoint signs and curvature. Exactly three
stationary points exist: two minima and one saddle in the plane. Rational
objective enclosures show a strict gap between the minima, establishing the
unique true global minimizer near (-1.1853025702,1.1853025702), J≈-1.4991448256.
The exact minimizer is defined by its isolated root, not a floating-point decimal.

This is a rigorous computer-assisted argument, not a formally verified proof
assistant development. Independent reviewers should concentrate on the analytic
reduction/tails and the correctness of the interval implementation and exported
cover. No floating-point sign or dense-grid count supports a proof decision.
The proof refuses unresolved cells/signs/value orderings rather than labelling
them complete. `verify_cover` also rechecks serialized proof CSV data.

## What was executed and compared

**116 tests passed, 0 failed, 0 skipped**: 87 original tests plus 29 additions.
`python scripts/run_all.py` completed all experiments and the proof, generating
6 figures and 28 CSVs. VALIDATION.md contains actual commands and environment.

Compared with baseline f21121f:

- All 26 original CSV files and all original manifest run settings/results are
  exactly unchanged, not merely equal at rounded display precision.
- Five figures are identical. Only the 1(c) contour legend changes to refer to
  the proven global minimum. Two separate proof CSVs were added.
- Counts remain Newton/BFGS 5/12; ordinary GD 138 to the lower root;
  gamma=.1,.3,.7 64/36/79 to the upper root. More momentum is not uniformly
  faster, and this comparison does not concern convergence to the same root.

REPORT_SNIPPETS.md contains literal source excerpts for Newton/Armijo,
expanded inverse BFGS and the shared momentum loop. A test checks excerpt
fidelity; omitted setup, helper behavior and status/recording roles are stated.
The important fallback and curvature checks are visible rather than hidden by
an invented shorter algorithm. CODE_MAP.md and REPORT_CODE_GUIDE.md map them.

## Convention audit and remaining work

REVIEW_REQUIRED.md classifies details as (1) original brief, (2) lecture
materials, (3) explicit implementation convention, or (4) still unproved if a
stronger claim is desired. Residual-norm stopping is lecture-supported;
strict equality handling, zero initial memory, objective-Armijo constants,
optional fallback, curvature safeguards and unprescribed sampling controls
are retained/disclosed conventions. They are not known bugs needing changes.

Checks confirm at most 200 updates, no hidden line search/fallback in 1(e),
recorded Newton fallback reasons, fresh zero momentum memory, the printed
2J.T F recurrence and no altered inverse-BFGS update. A new test proves 1(e)
bypasses Armijo entirely, even when Armijo might otherwise return alpha=1.

No required algorithmic clarification remains blocking. An exhaustive Exercise 2
root/minimum theorem is not supplied; its tasks' apparent/graphical estimates
remain numerical. The separately written report must explain conventions,
provide personal derivations/listings/references, fairly describe the new
computer-assisted proof and fit within 15 pages. No complete report is supplied.

## Review priorities

1. GLOBAL_MINIMUM_ANALYSIS.md: expansion of shifted product; strict v² increase;
   reduced derivatives; existence and analytic tail exclusion; mapping of
   reduced curvature to two-dimensional classifications.
2. `global_analysis.py`: exact arithmetic and outward rounding; Taylor remainder
   bounds; zero-free/unique-root conditions; complete coverage; strict rational
   objective ordering. Inspect both new CSVs and the fail-closed tests.
3. `part_c_minima.run`: numerical data is preserved; proof is separate;
   plotted global point must match the isolated root.
4. `experiments.py`: generated conclusions separate proofs and observations,
   include no universal Newton/momentum speed or basin guarantees, and preserve
   all existing experimental settings.
5. REPORT_SNIPPETS.md: check the literal excerpts remain appropriate report
   listings and the stated omitted helpers have no hidden numerical behavior.

REVIEW1_CHANGES.md records all substantive modifications and reproducibility
commands. Reproduce tests/experiments from the project directory with
`python -m pytest -q` and `python scripts/run_all.py` after README installation.
