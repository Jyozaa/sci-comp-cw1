# Verification actually performed — Review 1

Verified on **3 November 2026**, against baseline commit
`f21121fb07179b0de08231b30a846aa1cce9a214`. Environment: CPython 3.13.5,
macOS ARM64, NumPy 2.5.4, Matplotlib 3.11.2, pytest 9.2.0. Dependencies and
prescribed experiment settings were not changed. Other operating systems and
Python versions have not been directly tested.

## Actual test execution

```text
.venv/bin/python -m pytest -q
........................................................................ [ 62%]
............................................                             [100%]
116 passed in 1.59s
```

**116 passed, 0 failed, 0 skipped:** the original 87 tests are byte-for-byte
unchanged, with 25 reduction/interval-proof tests and 4 reporting/convention
tests added. Warnings remain errors in pytest configuration. Existing coverage
of analytical derivatives, quadratic Newton/BFGS behavior, expanded inverse
update/secant conditions, Armijo, fixed steps, finite failures, classification,
ordinary/momentum hand steps and stopping boundaries is preserved.

Additional checks exercise:

- Orthogonal transformation/inverse, norm preservation and equality with the
  original objective at independent off-line points.
- Reduced j, j', j'' against the original objective/gradient/Hessian and central
  finite differences; strict increase with v² and sign/tail inequalities.
- Exact rational interval operations, outward rounding and Taylor enclosures.
- Full root cover, exactly three simple roots, curvature classification,
  rational value separation and agreement with the original numerical search.
- Fail-closed proof behavior on depth exhaustion, missing cells and corrupted
  bounds. Exported CSV proof cells can be reloaded and checked exactly.
- All nine Python blocks in REPORT_SNIPPETS.md are literal contiguous excerpts
  from their cited source files, rather than shortened substitute algorithms.
- 1(e) never calls Armijo, using a monkeypatch that would raise on any call.
- Ordinary/momentum descent permit exactly 200 updates on nonvanishing test
  residuals, retain 201 history rows and report maximum-iteration failure.

## Actual experiment execution and output checks

Ran `.venv/bin/python scripts/run_all.py` after the changes. It completed every
required numerical experiment and the supplementary proof, producing all six
300 DPI figures, **28 CSVs** and the updated RESULTS.md/manifest. No solver or
prescribed numerical setting was changed to obtain these results.

The new proof produces a complete **40-cell rational cover** of [-2,2] with
three unique-root cells, no gaps and no extra roots in the analytically excluded
tails. Exact rational value bounds distinguish the two minima. The proof CSV
was separately reloaded with csv.DictReader and reverified using verify_cover.
The mathematical reduction, coercivity, finite bound, classification and Taylor
remainder justifications are in GLOBAL_MINIMUM_ANALYSIS.md; test success alone
is not being presented as that argument.

Compared actual output hashes with the committed baseline, not just approximate
printed numbers:

- **All 26 original CSVs are byte-for-byte unchanged**, covering every original
  iterate, convergence status/count and summary/classification result.
- **Five original PNGs are byte-for-byte unchanged.** Only
  `ex1_part_c_contours.png` changes: its global-point legend now refers to the
  independently supplied proof. The numerical data and plotted positions are
  unchanged. This updated figure was visually inspected.
- Two supplementary CSVs were added: `ex1_part_c_interval_cover.csv` and
  `ex1_part_c_reduced_stationary_points.csv`. Their tighter root approximations
  do not replace the original Newton/multistart rows.
- Every original manifest experiment entry is exactly equal to its baseline
  entry. `1c_global_analysis` is a separate new entry; output hashes/inventory
  include the two new files and changed legend.
- Recomputed SHA-256 for all 34 figure/table outputs and checked every manifest
  digest. Reopened/decoded all six PNGs and verified approximately 300 DPI.
- Byte-compared original objective/model files, Newton/BFGS/descent/momentum
  solvers, fixed-comparison modules, common utilities and original test files
  with the baseline: unchanged. Only the 1(c) postprocessing/legend and results
  generator were extended alongside the new supplementary module.

Original counts remain: 441 converged multistart runs, two minima and one saddle;
1(e) Newton/BFGS 5/12 updates; ordinary GD 138 updates to the lower root;
momentum .1/.3/.7 64/36/79 updates to the upper root. Required runs meet their
stated tolerance. The new proof now establishes why the Exercise 1 enumeration
is complete; the old grid itself did not.

The package was rebuilt/reinstalled for verification after adding the module.
All nine installed numerical module entry points were run with `-W error`
from a temporary directory outside the repository. All exited successfully;
their generated files, including the new proof tables, byte-match the current
results. Source syntax and all local Markdown links also passed final QA.
Source-tree tests/run_all always use the current source. README documents
reinstalling before standalone installed commands after edits.

## Initial baseline and remaining limits

The initial 2 November validation passed 87 tests and reproduced the original
6 figures/26 tables in an isolated directory. Its early BFGS zero-budget
regression was fixed before the baseline commit; no new core algorithm bug was
found or patched during Review 1. The baseline was subsequently pushed at the
user's explicit request. **This revision has not been committed or pushed.**

The global argument is a rigorous computer-assisted proof with explicit exact
arithmetic and Taylor bounds, not a formally verified theorem-prover result.
Independent review of that new argument/implementation remains appropriate.
Exercise 2's exhaustive root/minimum count remains unproved; its apparent and
graphical estimates do not claim completeness. Convention classifications,
source distinctions and report obligations are in REVIEW_REQUIRED.md. A full
coursework report, personal derivations, references and 15-page layout remain
separate work. No complete report or submission was produced.
