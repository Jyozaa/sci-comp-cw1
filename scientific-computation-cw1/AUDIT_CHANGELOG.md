# Coursework scope and simplicity audit

Baseline: commit `89bcd30dad54507e926fff031e1c0735112f6e64`.
Audit date: 3 November 2026. No automatic commit or push is authorised.

## Pre-edit checklist — completed before any deletion or restructuring

The original seven-page brief was read first; its rendered pp.2–4 equations
and requirements were visually checked. The corrected Markdown/rendered
reference was used only to clarify notation. Black regions were ignored.
The supplied ZIP was read privately outside the repository. Style references:
Week 1 Unit 2 notebook (short Newton loop), Week 2 Units 4–6 and Unit 5 notebook
(errors, convergence, bisection/secant), Week 3 Units 7–9 and Unit 8 notebook
(Jacobians, NumPy arrays/solve, step-halving), Week 4 Units 10–11/notebook
(gradient/Hessian, 2J.T F, residual stopping and trajectories). Units 12–15
and the Unit 14 notebook concern ODE/PDE discretisation; those topics are not
needed here. The archive has WEEK1–4 and WEEK6 directories, with no separate
WEEK5 directory; no missing teaching content is inferred.

All current source modules, tests, scripts, documentation/output inventory and
six existing figures were inspected. Existing derivatives and prescribed
updates appear correct. This is a scope/readability revision, not a request
for new numerical methods.

Paths in this checklist are under `src/sc_cw1/`. "Listing" refers to source code
explicitly requested in the submitted PDF, not whether a function exists.

| Part | Original brief requirement (page) | Relevant teaching / required calculation | Listing? / figure? | Current implementation | Scope/complexity finding and simplification decision |
| --- | --- | --- | --- | --- | --- |
| 1(a) | Derive first-order optimality conditions for equation (1), p.2 | Basic differentiation; Unit 10 gradient/Hessian; implement J and analytical gradient | No / no | `exercise1/model.py` | Formulas correct and short: preserve. Retain basic finite-difference checks; remove dependency on generic result/validation machinery. Written derivation remains report work. |
| 1(b) | Newton H*d=-grad, exact/numerical Hessian, gradient-norm stopping, Armijo, pp.2–3 | Units 2,7,8,9,10,11; numpy.linalg.solve and step-halving | Yes / no separate figure | `exercise1/part_b_newton.py`, `common/utilities.py:armijo` | Correct updates obscured by dataclass/configuration helpers. Put Armijo beside Newton; use ordinary functions and dictionary/list results; retain finite/singular/descent/failure checks and disclosed optional -grad fallback. |
| 1(c) | Different starts in [-2,2]², count minima/identify global, objective contours, p.3 | Newton multistart, objective comparison, basic Hessian second-derivative classification | No / yes | `exercise1/part_c_minima.py`, `global_analysis.py` | Exact interval/Taylor/root-count machinery is outside scope: delete it. Keep numerical multistart, deduplication, minima/saddle distinction, lowest candidate and honest numerical global wording. Use the simple 2×2 Hessian determinant test. |
| 1(d) | Implement the printed inverse-BFGS method, p.3 | Explicitly required by brief even without a detailed notebook implementation | Yes / no separate figure | `exercise1/part_d_bfgs.py` | Preserve expanded inverse formula. Remove unused skip policy and excess configuration; keep B0=omega I, direct matrix products, denominator/finite checks and explicit failure. |
| 1(e) | Plot gradient norms; x0=(-2,1), fixed alpha=1, omega=.01, p.3 | Required Newton/BFGS; Unit 5 semilog convergence plots | No / yes | `exercise1/part_e_comparison.py` | Correct settings: preserve. Fixed branches must bypass Armijo/fallback. Record initial point and histories with simple arrays; consolidate summaries. |
| 2(a) | Derive four Jacobian entries, p.4 | Units 7–8 Jacobian; Unit 10 phi=F1²+F2² and 2J.T F | No / no | `exercise2/model.py` | Residuals/Jacobian/gradient correct: preserve. Supplementary residual Hessian/rate-analysis code is unnecessary: remove. |
| 2(b) | Plot phi isocontours on [-2,2]² and count apparent minima, p.4 | Unit 8/11 meshgrid/contour examples | No / yes | `exercise2/part_b_contours.py` | Grid plot required; Newton refinement, residual-Hessian classification and repeated-resolution study unnecessary. Keep the plot and simple neighbouring-grid estimates; do not call approximate nonzero residuals exact roots. |
| 2(c) | Draw F1=0 and F2=0; estimate intersections graphically, p.4 | Unit 8/11 plotting; basic sampled sign changes/interpolation (Units 4–6) | No / yes | `exercise2/part_c_intersections.py` | Generic all-segment intersection framework and additional bisection table exceed this graphical task. Keep the actual circle/cosine plot and simple sampled/interpolated crossing estimates; no iterative refinement. |
| 2(d) | Fixed lambda=.05, x0=(0,-1), tol=1e-6, maxit=200; plot path, count/destination/slow convergence, p.4 | Unit 10/11 direct residual gradient descent; Unit 4 convergence | No / yes | `exercise2/part_d_gradient_descent.py` | Preserve recurrence and lecture-supported residual norm; put 2J.T F visibly in the loop. Replace result class/shared evaluation framework with lists/dictionary; retain finite failures and exact update budget. Explain slow progress from history, not extra Hessian-eigenvalue experiments. |
| 2(e) | Implement printed momentum recurrence; show modifications, p.4 | Explicitly required by brief; extends Unit 10/11 loop | Yes / no separate figure | `exercise2/part_e_momentum.py`, shared loop in 2(d) | Keep one short shared loop with gamma=0 for ordinary descent, documented zero previous direction, and visible d=gamma*d+g. Remove unused custom-memory machinery and duplicate demonstration outputs. |
| 2(f) | Same settings; gamma=.1,.3,.7; plot trajectories/explain change, p.5 | Brief's momentum; Unit 11 trajectory plots | No / yes | `exercise2/part_f_comparison.py` | Preserve independent runs and final residual/status. Keep direct trajectory plots and an objective-rise count relevant to oscillation; remove unrelated path-length/step-reversal diagnostics and duplicate summaries. |

The six pre-edit figures use correct functions/settings. The 1(c) legend's
proof claim will be removed; 2(b/c) annotations will become explicitly graphical
estimates after removing refinement; the momentum comparison can use one clear
panel. All figures will be regenerated and inspected after simplification.

## Decisions before implementation

- Keep question-based modules and correct mathematical model expressions.
- Remove classes, generic result/evaluation/configuration systems, exact
  arithmetic proof, supplementary refinement/rate/resolution experiments,
  output hashes/manifest and duplicate demonstration files.
- Keep ordinary NumPy functions, direct solves/matrix products, simple lists,
  dictionaries, basic CSV export and a small pytest suite.
- Consolidate documentation into README, CODE_MAP, RESULTS, REPORT_SNIPPETS,
  REVIEW_REQUIRED and this changelog. Record the actual changes/deletions and
  verification below after execution; do not claim a test run in advance.

## Implemented changes

Paths below are relative to this project. The deletion inventory at the end
names every deleted file. “Same” means the required solver calculations agree
exactly with the baseline under the retained settings, not that CSV bytes or
schemas are unchanged.

| File(s) changed | Part | What and why | Mathematical behaviour | Numerical output effect | Purpose/category |
| --- | --- | --- | --- | --- | --- |
| `src/sc_cw1/common/utilities.py` | All | Replace SolverResult dataclass, evaluation/shape/error wrappers, generic differentiation and plotting with basic CSV, directory, summary and figure functions. Move Armijo beside Newton. Histories are ordinary dictionaries of arrays/lists. | Same required solver updates; public result API intentionally simpler. Unused arbitrary-input/configuration support removed. | Same required histories; fewer columns, LF CSV newlines. | Simplified structure; removed unnecessary framework. |
| `src/sc_cw1/exercise1/model.py` | 1(a/b) | Preserve J, gradient, analytical Hessian and array formula; direct NumPy conversion. Replace generic/scaled central-difference helper with a visible two-column fixed-step numerical Hessian. | Analytical formulas unchanged. Optional numerical-Hessian approximation uses h=1e-5 rather than a coordinate-scaled h; default experiments use the analytical Hessian. | No required result changes. | Simplified structure. |
| `src/sc_cw1/exercise1/part_b_newton.py` | 1(b/c/e) | Put ordinary Armijo and direct Newton loop together; keep solve, finite/condition checks, recorded optional -gradient fallback and explicit failures. Remove generic result/config machinery. | Same required directions and steps. Fallback remains off by default, on explicitly in 1(b/c), off in 1(e). | Same required results; 1(b) standalone prints one example instead of saving four duplicate demonstration runs. | Simplification; remove duplicate output. |
| `src/sc_cw1/exercise1/part_c_minima.py` | 1(c) | Remove proof call/import; retain 441 starts, simple distance clustering and comparison. Use the elementary 2×2 Hessian determinant/H11 test rather than eigenvalue reporting; relabel lowest point as numerical global candidate. | Same points/classifications for this experiment. No exhaustive root/global claim. Near-degenerate Hessians remain inconclusive. | All 441 final outcomes, counts, objectives and cluster representatives unchanged; no proof tables. | Scope compliance; simplification; graph correction. |
| `src/sc_cw1/exercise1/global_analysis.py` and its proof document/test/tables (deleted) | 1(c) | Remove rational interval arithmetic, Taylor remainder bounds, reduced-variable root isolation and exhaustive/global proof infrastructure. | Supplementary proof removed; numerical optimisation unaffected. | Two proof CSVs removed. | Removed beyond-scope methods, as requested. |
| `src/sc_cw1/exercise1/part_d_bfgs.py` | 1(d/e) | Direct loop/dictionary; preserve expanded inverse formula and B0=omega I. Remove configurable skip/reset-style machinery; fail plainly on unsafe curvature. Reuse objective Armijo only when enabled. | Prescribed inverse update unchanged; no damping, skip or denominator replacement. | Required history unchanged; standalone prints one example instead of duplicate files. | Simplification; remove unused configuration/output. |
| `src/sc_cw1/exercise1/part_e_comparison.py` | 1(e) | Use simple result arrays for plot/history; explicitly disable line search and fallback with required start, unit step, omega. | Same. | Same curves/counts, simpler labels; summary consolidated. | Simplification; graph/documentation clarity. |
| `src/sc_cw1/exercise2/model.py` | 2(a/b/d/e) | Preserve F, Jacobian, phi, gradient/grid formulas; remove supplementary residual Hessian. | Required formulas unchanged; supplementary Hessian API removed. | No required solver changes. | Removed unrequested analysis; simplified conversion. |
| `src/sc_cw1/exercise2/part_b_contours.py` | 2(b) | Keep full square contours and straightforward eight-neighbour grid comparison. Remove Newton refinement, residual-Hessian classification and multiple-resolution study. Make finite-grid labels explicit, improve colourbar readability, keep annotations inside axes. | Apparent grid minima replace refined roots for this graphical task. | Two grid locations (.35,.76), (1.38,-.93), with nonzero residuals honestly reported; count unchanged. | Scope/simplicity; graph update. |
| `src/sc_cw1/exercise2/part_c_intersections.py` | 2(c) | Replace generic all-segment intersection framework and bisection refinement with sampled circle/cosine plots and a single linear interpolation at each sign change. | Graphical estimates only; no iterative root refinement. | Graphical coordinates differ by at most 2.53e-7 from old graphical table; residuals remain around 1e-6. Refined table removed. | Scope/simplicity; graph update. |
| `src/sc_cw1/exercise2/part_d_gradient_descent.py` | 2(d/e) | One direct loop with visible 2J.T F, fixed step, residual stopping and list histories; three comments identify the momentum additions. Keep finite failure status and exact update budget. | Same ordinary/momentum updates, including initial/final-state checks; no gradient-based false convergence. | Same required histories; smaller CSV schema. | Simplified structure; graph labels. |
| `src/sc_cw1/exercise2/part_e_momentum.py` | 2(e/f) | Small wrapper chooses gamma and calls the shared loop with explicit keyword arguments. Remove unused custom initial-direction option; fresh zero memory is visible. | Required recurrence and initial memory unchanged. | Same .3 example; no duplicate standalone momentum CSV/summary. | Simplification; remove unrequested option/output. |
| `src/sc_cw1/exercise2/part_f_comparison.py` | 2(f) | Three independent prescribed runs; one clear trajectory panel, distinct markers, start/final legend outside the path. Remove supplementary panel/diagnostics. | Same. | Same three histories/paths/counts; figure layout changes. | Simplification; graph update. |
| `src/sc_cw1/experiments.py` | All | Run just the six figure-producing tasks (also exercising required solvers); collect one comparison summary and generate RESULTS from computed values. Remove proof calls, duplicate demonstrations, eigenvalue/rate diagnostics, resolution study, environment/hash manifest. | Required experiments unchanged; supplementary runs removed. | Six figures and eleven CSVs instead of 28 CSVs plus manifest. New `comparison_summary.csv` replaces per-part summaries. | Removed unnecessary functionality; simplified reporting. |
| `results/figures/ex1_part_c_contours.png` | 1(c) | Regenerated from current model, distinct minimum/saddle markers and numerical global-candidate legend. | None. | Same contour/point data; revised wording. | Graph update. |
| `results/figures/ex1_part_e_newton_vs_bfgs.png` | 1(e) | Regenerated semilog norms with fixed-unit-step/start/omega title and statuses. | None. | Same histories. | Graph update. |
| `results/figures/ex2_part_b_residual_contours.png` | 2(b) | Regenerated full-square phi contours with apparent grid minima and readable levels/annotations. | See 2(b) above. | Graphical locations replace refined-root labels. | Graph update. |
| `results/figures/ex2_part_c_intersections.png` | 2(c) | Regenerated circle/cosine intersections with three-decimal graphical labels inside axes. | See 2(c) above. | Tiny graphical-estimate changes. | Graph update. |
| `results/figures/ex2_part_d_gradient_descent.png` | 2(d) | Regenerated required fixed-step trajectory with start/final/count/status. | None. | Same path. | Graph update. |
| `results/figures/ex2_part_f_momentum_comparison.png` | 2(f) | Regenerated three paths in one panel, readable marker/legend distinction. | None. | Same paths; simpler layout. | Graph update. |
| `results/tables/ex1_part_c_multistart.csv`, `ex1_part_c_stationary_points.csv` | 1(c) | Keep outcomes/representatives, fallback counts and basic Hessian test; drop repeated configuration/proof/eigenvalue fields. | Same classification/results. | All common columns exactly match baseline; determinant/H11 replace eigenvalues. | Output simplification. |
| `results/tables/ex1_part_e_newton_history.csv`, `ex1_part_e_bfgs_history.csv` | 1(e) | Keep k, point, J, gradient norm, alpha, step kind/status. | Same. | All seven common columns exactly equal; 6/13 rows. | Output simplification. |
| `results/tables/ex2_part_b_minima.csv`, `ex2_part_c_graphical_intersections.csv` | 2(b/c) | Keep graphical estimates and actual residuals, without claiming refined roots. | As above. | Intentional geometry/schema/precision changes. | Scope compliance/output simplification. |
| `results/tables/ex2_part_d_gradient_descent_history.csv`, `ex2_part_f_gamma_0.1_history.csv`, `ex2_part_f_gamma_0.3_history.csv`, `ex2_part_f_gamma_0.7_history.csv` | 2(d/f) | Keep iteration points, phi, norms, fixed step, kind/status. | Same. | All eight common columns exactly equal; 139/65/37/80 rows. | Output simplification. |
| `results/tables/comparison_summary.csv` (new) | 1(e), 2(d/f) | One short summary of counts, statuses, final points/norms and observed objective increases. | No additional experiment. | Consolidates six runs; objective increases 0/0/0/0/1/28 in row order. | Output consolidation. |
| `requirements-lock.txt`, `results/manifest.json` (deleted) | All | Drop machine-specific dependency freeze and generated hash/config/runtime manifest; retain small direct requirements and normal package metadata. | None. | No manifest generated. | Remove unnecessary infrastructure. |
| Duplicate CSVs listed below (deleted) | 1(b/d/e), 2(b/c/d/e/f) | Remove repeated examples/summaries, refined-root/resolution/rate diagnostics. Essential histories and six figures remain. | No required solver altered. | 18 old CSVs removed, one new summary added: 28 to 11. | Remove unrequested/duplicate outputs. |
| `tests/test_exercise1.py` | 1(a–e) | Compact independent derivative, Hessian, Armijo, fixed-step and full multistart tests. | Test coverage only. | No stored-answer forcing. | Changes to test coverage. |
| `tests/test_exercise2.py` | 2(a–f) | Compact derivative, hand-computed update, residual stopping, exact budget, required coefficients and graphical-precision tests. | Test coverage only. | No required output alteration. | Changes to test coverage. |
| `tests/test_numerical_methods.py` | 1(b/d), 2(d/e) | Small quadratic/Newton/BFGS secant examples and meaningful singular/nonfinite/curvature/line-search failures. | Test coverage only. | No production changes. | Changes to test coverage. |
| `tests/test_review1_global_analysis.py`, `tests/test_review1_reporting.py` (deleted) | 1(c), reporting | Remove proof and framework/report-schema tests; retain essential budget/fixed-step checks in the three compact test modules. | Test coverage only. | Old 116-test suite consolidated to 32 cases; discarded tests target removed scope/API. | Remove proof/framework tests; consolidate coverage. |
| `README.md` | All | Current installation/reproduction, conventions, question commands and ordinary-dictionary API; six-document navigation. | None. | Correct six-figure/eleven-table inventory. | Updated documentation. |
| `CODE_MAP.md` | All | Map all eleven parts and every mathematical dependency, including shared Armijo and momentum lines. | None. | Accurate current outputs/listings. | Updated documentation. |
| `RESULTS.md` | All | Regenerated by experiments with numerical-candidate wording, actual graphical precision, unchanged solver counts and observational conclusions. | No altered numerical answer. | Removes proof/refinement/rate claims and duplicate tables. | Updated generated documentation. |
| `REPORT_SNIPPETS.md` | 1(b/d), 2(e) | Replace stale class/config excerpts with literal current Newton/Armijo, expanded inverse BFGS and three momentum changes; identify all omitted bookkeeping. | No shortened replacement algorithm. | Current-source excerpts. | Updated documentation. |
| `REVIEW_REQUIRED.md` | All | Consolidate real numerical limitations/conventions, source support and written-report checks. | None. | Explicit numerical-only global claim and graphical accuracy. | Updated documentation. |
| `AUDIT_CHANGELOG.md` (new) | All | Preserve pre-edit checklist, source review, per-file changes/deletions and actual verification in this single handoff. | None. | Replaces duplicate validation/handoff documents. | Updated documentation. |
| Seven duplicate/supplementary Markdown documents listed below (deleted) | All | Consolidate source audit, validation, handoff, code-guide and previous-review prose into the six requested documents; remove proof text. | None. | No stale runtime/document links to removed files. | Documentation consolidation. |

## Removed methods and important retained choices

1. Removed from `exercise1/global_analysis.py`: exact rational interval
   arithmetic, Taylor-series remainder enclosures, one-dimensional interval
   root isolation/exhaustive cover and computer-assisted global-minimum proof.
   No replacement proof was introduced.
2. Removed from `exercise2/part_b_contours.py` / `exercise2/model.py`:
   supplementary Newton root refinement, residual-Hessian/eigenvalue
   classification and repeated-resolution comparisons.
3. Removed from `exercise2/part_c_intersections.py`: the generic pairwise
   segment-intersection framework and supplementary iterative bisection
   refinement. Bisection itself is taught; it was removed because 2(c) asks
   for graphical estimates, not because it is an impermissible technique.
4. Removed from `experiments.py`: extra local Hessian/eigenvalue convergence
   rate diagnostics and other unrequested reporting experiments. Eigenvalues
   and numerical analysis are not generally forbidden; these additions were
   unnecessary to answer the specific questions.
5. Removed generic result/validation/configuration infrastructure in
   `common/utilities.py` and solver modules. This is a software simplification,
   distinct from removal of an advanced mathematical method.

No confirmed core algorithmic bug was found or repaired. Working objective,
derivative, Newton, BFGS and descent/momentum formulas were preserved.
A newly written test initially expected a 5×5 start grid to find the saddle;
it did not. The test was corrected to exercise the actual 21×21 experiment,
which does find it. This is also a reminder that finite-grid coverage is not
an exhaustive stationary-point argument.

The retained numerical safeguards are direct and visible: singular/conditioned
Hessian handling, optional disclosed -gradient fallback, non-descent/Armijo
failure, finite-value checks, unsafe curvature failure and non-convergence
status. `np.errstate` suppresses expected overflow warnings while explicit
finite checks reject invalid states; it does not manufacture accepted steps.
The BFGS inverse update, fixed alpha=1 comparison and printed momentum
recurrence have not been replaced. Dependency pins, package metadata and the
small launcher are unchanged. No SciPy, automatic differentiation or external
solver was added.

## Actual verification and numerical comparison

Executed after the final mathematical/plotting edits, using local Python
3.13.5 and the existing environment:

- `python -m pip install . --no-deps --force-reinstall`: succeeded after
  clearing generated build/egg-info caches. This rebuild avoids packaging a
  stale deleted module.
- `python -m pytest -q`: **32 passed in 0.48s**, no failures.
- `python scripts/run_all.py`: **exit 0**, six required figures and eleven
  CSVs regenerated. This executes 441 multistart Newton runs, the fixed-step
  Newton/BFGS comparison, residual contours, graphical curves, ordinary
  descent and the three fresh momentum runs. 1(a)/2(a) formulas are exercised
  throughout and independently checked by derivative tests.
- All nine independent question-module commands in README were executed from
  a separate temporary working directory against the newly installed package:
  **nine exit 0, no stderr**. The installed package has no `global_analysis`
  module. They do not depend on private source PDFs/notebooks or repository
  working-directory imports.
- CSVs were compared directly against `git show` of baseline commit
  `89bcd30`: all common columns are **exactly equal** for the six required
  iteration histories, all **441** multistart outcomes and all **three**
  stationary representatives. This includes coordinates, objective values,
  norms, update counts, statuses, step kinds and fallback counts where shared.
- Final review: `git diff --check` passed; all ten Python blocks in
  REPORT_SNIPPETS were verified as literal contiguous source excerpts; all
  local Markdown links resolve; exactly the six requested Markdown files
  remain. Searches of runnable code/tests/current guidance found no stale
  proof, removed-helper or removed-document references. HEAD is unchanged
  and the staging area is empty.
- The new CSV schema deliberately omits repeated controls, messages,
  eigenvalues and per-step diagnostics; it uses LF newlines. Bytewise CSV
  equality is therefore neither expected nor claimed.
- The two 2(b) estimates intentionally have grid precision (.01 spacing),
  replacing supplementary refined roots. The graphical 2(c) interpolation
  differs by at most 2.53e-7 per coordinate from the old graphical table;
  residual norms are approximately 1.52e-6 and 2.15e-6. Those small graphical
  changes do not affect any optimisation trajectory or convergence count.

| Experiment | Updates / result | Comparison with baseline |
| --- | --- | --- |
| 1(c), 441 starts | 441 converged; two minima (clusters 205/232), one saddle (4) | Same; lowest (-1.185302570,1.185302570), J=-1.499144826 |
| 1(e), Newton | 5, converged | Exactly same required history |
| 1(e), BFGS | 12, converged | Exactly same required history |
| 2(d), ordinary descent | 138, lower root | Exactly same required history |
| 2(f), gamma=.1 | 64, upper root | Exactly same required history |
| 2(f), gamma=.3 | 36, upper root | Exactly same required history |
| 2(f), gamma=.7 | 79, upper root | Exactly same required history |

No results are hardcoded into the algorithms. Tests of known examples and
required settings remain, while the baseline CSV comparison is an independent
read-only audit action rather than a new experiment or framework in the repo.

## Figure inspection

All six regenerated PNGs were opened and visually inspected, both individually
and together. Each file was decoded and its metadata checked: **six readable
PNGs at 300 DPI** (PNG metadata rounds to 299.9994). The contour/curve figures
cover [-2,2]²; the trajectory views include all their prescribed paths. Labels,
functions, starts, parameters, legends and stationary-point distinctions match
the brief. No proof or exhaustive-root claim appears in a figure.

- 1(c): original J; red-star lowest candidate, distinct second minimum and
  saddle markers; neither saddle nor fallback is described as a local minimum.
- 1(e): actual gradient norms on a logarithmic vertical axis, including x0;
  title/legend disclose fixed steps, start, omega and 5/12 updates.
- 2(b): phi contours on full square with readable levels and apparent grid
  minima. The right-hand annotation was moved inside the axes after inspection.
- 2(c): correct unit circle centred at (1,0), cosine and both graphical
  crossings; three-decimal labels fit inside the frame.
- 2(d): actual .05-step trajectory from (0,-1) to the lower root, showing 138
  updates and convergence, with start/final markers.
- 2(f): all three prescribed trajectories, independent common start, distinct
  colours/markers and final points; legend moved below the axes so it does not
  cover the paths. The .7 oscillation remains visible.

## Compliance and remaining report work

All eleven questions have executable calculations mapped in CODE_MAP.
The required figures/listings/settings are retained. The only substantive
mathematical limitation is 1(c): the lowest minimum is supported numerically,
**not established as the true global minimum by a rigorous proof** in this
scope-limited version. Treat that part as partial if a formal global proof is
required; ask the lecturer about the expected strength of numerical evidence.
No claim of complete mathematical correctness is made on that unresolved point.

The written first-order/Jacobian derivations, report narrative, code selection,
references and final single-PDF layout/page count still require the student's
work. No final report was requested or generated. REVIEW_REQUIRED records
retained conventions (including residual stopping and zero initial momentum),
which can affect counts/basins and must not be called explicit brief settings.
No source material was copied into the repository or obscured text inferred.
Only the six requested Markdown documents remain. No commit, staging or push
was performed during this audit; the user will review and push separately.

Reproduce from this directory with the README installation commands, then:

```bash
source .venv/bin/activate
python -m pytest -q
python scripts/run_all.py
```


## Deleted files

These historical paths are recorded only as the deletion inventory.

- `COMPLETION_REPORT.md`
- `GLOBAL_MINIMUM_ANALYSIS.md`
- `REPORT_CODE_GUIDE.md`
- `REVIEW1_CHANGES.md`
- `REVIEW_HANDOFF.md`
- `SOURCE_AUDIT.md`
- `VALIDATION.md`
- `requirements-lock.txt`
- `results/manifest.json`
- `results/tables/ex1_part_b_newton_0.csv`
- `results/tables/ex1_part_b_newton_1.csv`
- `results/tables/ex1_part_b_newton_2.csv`
- `results/tables/ex1_part_b_newton_3.csv`
- `results/tables/ex1_part_b_summary.csv`
- `results/tables/ex1_part_c_interval_cover.csv`
- `results/tables/ex1_part_c_reduced_stationary_points.csv`
- `results/tables/ex1_part_d_bfgs_history.csv`
- `results/tables/ex1_part_d_summary.csv`
- `results/tables/ex1_part_e_summary.csv`
- `results/tables/ex2_local_convergence_diagnostics.csv`
- `results/tables/ex2_part_b_grid_candidates.csv`
- `results/tables/ex2_part_b_resolution_check.csv`
- `results/tables/ex2_part_c_refined_intersections.csv`
- `results/tables/ex2_part_d_summary.csv`
- `results/tables/ex2_part_e_momentum_history.csv`
- `results/tables/ex2_part_e_summary.csv`
- `results/tables/ex2_part_f_summary.csv`
- `src/sc_cw1/exercise1/global_analysis.py`
- `tests/test_review1_global_analysis.py`
- `tests/test_review1_reporting.py`
