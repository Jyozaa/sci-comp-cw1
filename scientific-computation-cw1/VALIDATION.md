# Verification actually performed

Verified on 2 November 2026, using CPython 3.13.5 on macOS ARM64,
NumPy 2.5.4, Matplotlib 3.11.2 and pytest 9.2.0. Full runtime/test dependency
versions are in requirements-lock.txt and results/manifest.json. Other Python
versions and operating systems have not been directly tested.

## Tests

Actual command:

```text
.venv/bin/python -m pytest -q
........................................................................ [ 82%]
...............                                                          [100%]
87 passed in 0.33s
```

Final result: **87 passed, 0 failed, 0 skipped**. Warnings are errors in the
pytest configuration. The suite covers:

- Exact values and independent objective-to-gradient central differences.
- Analytical Hessian against central gradient differences **and direct second
  differences of the scalar objective**, plus Hessian symmetry.
- Independent Exercise 2 Jacobian, residual-gradient and residual-Hessian checks.
- Newton on a known quadratic in one step; inverse BFGS on known quadratics;
  agreement of the printed expanded update with the compact inverse form;
  secant equation, symmetry and positive definiteness on test data.
- A guard proving BFGS does not call a linear-system solve.
- Actual accepted objective-Armijo inequalities, rejected non-descent and
  exhausted searches, overflowed-trial recovery and numerical-Hessian use.
- Prescribed fixed unit steps, initial point and inverse scaling in 1(e).
- Classification/deduplication, including excluding failed runs and a
  nonstationary point; distinction between minima, saddles and maxima.
- Hand-computable ordinary and momentum updates, factor 2 in 2J.T F, zero/custom
  initial momentum, fresh memory across runs and all three prescribed gammas.
- Exact iteration budgets, initial/final-update convergence, strict tolerance
  equality, nonroot stationary residuals, NaN/Inf/overflow, failed-trial history
  preservation, singular/ill-conditioned Hessians and unsafe curvature.
- Graphical-segment intersection estimates, separately refined residuals,
  grid/scalar agreement and apparent-minimum curvature/residual checks.

An initial test caught BFGS reporting a zero-budget, already-optimal initial
point as nonconverged. The solver now checks that case, and the regression
passes. No failing assertion was weakened to conceal this bug.

## Execution and generated artifacts

- Actually ran `.venv/bin/python -W error scripts/run_all.py`, completing every
  numerical part. Exactly **6 required PNGs and 26 CSV tables** were produced,
  plus RESULTS.md and results/manifest.json. Required coursework runs converged.
- Ran all **nine independent numerical module CLIs** with `-W error`, from a
  temporary working directory outside the repository and with a separate output
  directory. All exited successfully. Imported both analytical model modules,
  shared utilities and the experiment orchestrator outside the source tree.
- Built and installed the package with `python -m pip install . --no-deps`.
  An editable-install issue specific to this macOS environment marked its .pth
  file hidden, causing Python to ignore it. Standard package installation
  resolved this; README uses that verified setup. Tests and run_all use the
  current source directly, while standalone installed module commands require
  reinstalling after source edits.
- Repeated the **entire all-experiment script from a fresh temporary directory**.
  All 6 PNGs, all 26 CSVs, and RESULTS.md were byte-for-byte identical to the
  delivered outputs on this environment. Cross-platform byte identity is not
  promised, especially for PNGs/fonts and BLAS floating-point details.
- Recomputed SHA-256 for all 32 figure/table outputs; every digest matched the
  manifest. Reopened/decoded each PNG, verified dimensions of at least 1500×1000
  pixels and approximately 300 DPI, and visually reviewed all six final plots.
  Adjusted the residual colorbar to show powers of ten, kept intersection labels
  inside the plot, and moved the momentum legend away from the trajectories.
- Independently audited the saved histories: **413 finite updates** have
  sequential iteration numbers, correct saved objective values and the expected
  incoming update. Checked Newton linear equations, normal Armijo inequalities,
  prescribed unit/.05 steps, residual momentum recurrence and final stopping
  statistics. Checked all 441 multistart statuses and that ordinary Newton plus
  fallback counts equal each run's update count.
  Reconstructed both saved BFGS runs with the independent compact inverse
  formula, checking their search directions and recorded curvature values.
- Checked source syntax, local documentation links, required project files,
  and absence of `eval()` and bundled source PDFs/notebooks. Source identities
  and equation checks are documented in SOURCE_AUDIT.md.

## What these checks do not establish

They do not prove global optimality or exhaustive stationary-point/root counts.
They do not establish the intended interpretation of the unspecified Exercise 2
stopping statistic or initial momentum. They do not replace the written
analytical derivations or the assessment report. REVIEW_REQUIRED.md enumerates
these limits, safety conventions, exact source ambiguities and manual checks.

No GitHub commit, push or submission was performed.
