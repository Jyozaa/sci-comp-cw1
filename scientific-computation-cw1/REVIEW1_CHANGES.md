# Review 1 change record

Revision date: 3 November 2026. Baseline: `f21121f` on main. The independent
review found no confirmed core algorithmic defect. This revision does not
restructure the project or change required numerical settings or algorithms.
The changes are local and uncommitted; nothing was pushed automatically.

## Substantive changes by file

| File | Modification and reason |
| --- | --- |
| `src/sc_cw1/exercise1/global_analysis.py` (new) | Orthogonal/reduced models and derivatives; exact rational interval/Taylor proof of root completeness and strict minimum ordering; fail-closed checker; exports proof tables. Separate from all existing optimisers. |
| `GLOBAL_MINIMUM_ANALYSIS.md` (new) | Independently verified algebra; analytical derivatives, monotonicity, coercivity and explicit finite bounds; reproducible interval proof; exact root definition, classifications and value bounds; proof/numerical/implementation limitations. |
| `src/sc_cw1/exercise1/part_c_minima.py` | Runs the supplementary proof after unchanged multistart outputs, checks association with the numerical lowest point, adds proof to returned diagnostics and changes the contour legend. Does not modify the search, classification, tolerances or iterates. |
| `src/sc_cw1/experiments.py` | Updates the RESULTS.md generator with proof evidence and more careful convergence/basin explanations; adds a separate proof manifest entry. Original run settings stay unchanged. |
| `tests/test_review1_global_analysis.py` (new) | 25 regression checks covering reduction/derivatives, rational arithmetic, Taylor bounds, roots/order, fail-closed behavior and rechecking a CSV roundtrip. |
| `tests/test_review1_reporting.py` (new) | 4 checks for nine literal source excerpts, complete bypass of Armijo in 1(e), and exactly 200 permitted nonconvergent updates for ordinary/momentum descent. |
| `REPORT_SNIPPETS.md` (new) | Literal source excerpts for 1(b),1(d),2(e), including safeguards, stopping and the actual shared loop; explicitly identifies omitted setup/helpers. Not replacement algorithms or a full report. |
| `REVIEW_REQUIRED.md` | Replaces unresolved R1 with supplied rigorous computer-assisted proof; classifies every R2–R6 detail by original/lecture/convention/remaining mathematical verification and explains correctness/count/presentation impact. |
| `README.md`, `CODE_MAP.md`, `REPORT_CODE_GUIDE.md` | Map the supplemental proof and literal listings, update output count and remove obsolete unresolved-global wording. |
| `COMPLETION_REPORT.md` | Updates global status, current test/output counts and proof-data inventory; distinguishes current revision from the previously published baseline. |
| `SOURCE_AUDIT.md` | Records unchanged source hashes and repeat notebook/convention inspection; attributes the reduction idea to the independent reviewer. |
| `VALIDATION.md`, `REVIEW_HANDOFF.md` | Actual Review 1 execution evidence, exact baseline comparisons, remaining reporting work and new proof-review priorities. |
| `RESULTS.md` (generated) | Regenerated through the changed generator; distinguishes global proof, original numerical outcomes, and observations about convergence/basins. |
| `results/figures/ex1_part_c_contours.png` (generated) | Legend-only change referring to the proved global minimum; all plotted numerical data/positions preserved. |
| `results/tables/ex1_part_c_interval_cover.csv` (new) | All 40 rational proof cells, enclosures, signs and unique-root brackets. |
| `results/tables/ex1_part_c_reduced_stationary_points.csv` (new) | Separate tightly enclosed root/value data and supplementary approximations; original stationary-point table is not overwritten. |
| `results/manifest.json` (generated) | Adds proof settings/two files and updated contour hash. Every original experiment entry is unchanged. |

## Preserved implementation and numerical evidence

Byte comparisons with baseline confirm the objective/derivative model,
Newton, expanded inverse-BFGS, 1(e) fixed-unit-step comparison, residual model,
ordinary descent, printed momentum recurrence, comparison modules and shared
utilities are unchanged. All three original test files are unchanged.

All 26 original CSVs are byte-identical to the baseline. Five figures are
byte-identical; only the global legend changes in the sixth. No original
numerical coordinate, gradient/residual norm, status or iteration count changed.
All original manifest parameters/results compare exactly equal. The two new
proof tables are supplementary, not replacements for required experiment data.
There are now 6 figures and 28 CSVs.

The global-minimum conclusion changed because a new justified proof was
supplied, not because an algorithm or an answer was tuned. Approximate root
refinements in the new table may be more accurate than a multistart row (most
visibly the saddle); that is labelled and does not alter the original results.

## Verification

Actual final test result: **116 passed in 1.59s; 0 failed; 0 skipped**.
The all-experiment script was run successfully, all six figures exist/decode
at approximately 300 DPI, and all 34 figure/table hashes match the manifest.
The exported proof cover was separately loaded/rechecked. The changed contour
was visually inspected. See VALIDATION.md for commands and detailed limits.

```bash
cd scientific-computation-cw1
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_all.py
```

No complete coursework report, Git commit or GitHub push was made in this revision.
