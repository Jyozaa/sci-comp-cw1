# Scientific Computation Coursework 1

Reproducible Python implementations of the visible COMP5930M Exercise 1 and 2
requirements. The code uses NumPy, Matplotlib and pytest; no SciPy optimiser,
automatic differentiation, GPU, remote service or paid API is required.

All numerical sections are implemented. Review 1 adds a rigorous
**computer-assisted global-minimum proof** for Exercise 1(c), using analytical
reduction and exact rational root/value enclosures. See
[GLOBAL_MINIMUM_ANALYSIS.md](GLOBAL_MINIMUM_ANALYSIS.md). The original algorithms,
parameters and numerical histories are unchanged. The residual stopping criterion
is lecture-supported; initial momentum and safety choices are explicit
conventions classified in [REVIEW_REQUIRED.md](REVIEW_REQUIRED.md).

## Install and reproduce

Python **3.12 or newer** is required with the pinned dependencies. Development
and verification used CPython 3.13.5. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install . --no-deps
python -m pytest -q
python scripts/run_all.py
```

On Windows activate `.venv\Scripts\activate` instead. `requirements.txt` pins the
three direct dependencies. `requirements-lock.txt` also records transitive
versions used here; use it in place of `requirements.txt` for closer replication.
The package install makes `sc_cw1` importable. pytest and the all-experiment
script use the current source tree directly. After editing source, reinstall
with `python -m pip install . --no-deps --force-reinstall` before using the
independent module commands, so the installed package reflects those edits.

The all-experiment command takes seconds on a laptop. It regenerates the six
300 DPI PNGs under `results/figures/`, 28 CSVs under `results/tables/`,
`results/manifest.json`, and [RESULTS.md](RESULTS.md). The manifest records the
runtime, run parameters, statuses and output SHA-256 hashes. Different platforms
may produce small floating-point differences and different PNG hashes/fonts.
There is no random sampling. Source PDFs/notebooks are unnecessary at runtime.

Choose another results directory with:

```bash
python scripts/run_all.py --output /tmp/cw1-check/results
```

The computed `RESULTS.md` is written to the parent of that results directory.
Existing files with generated names there are overwritten. Independent question
commands update only their own outputs; regenerate the complete digest/manifest
with `run_all.py` after changing experiments.

## Run each question independently

After the package install, each command works from this directory:

| Part | Command |
| --- | --- |
| 1(a) | `python -c "from sc_cw1.exercise1.model import objective,gradient,hessian; print(objective([0,0]),gradient([0,0]),hessian([0,0]))"` |
| 1(b) | `python -m sc_cw1.exercise1.part_b_newton` |
| 1(c) | `python -m sc_cw1.exercise1.part_c_minima` |
| 1(d) | `python -m sc_cw1.exercise1.part_d_bfgs` |
| 1(e) | `python -m sc_cw1.exercise1.part_e_comparison` |
| 2(a) | `python -c "from sc_cw1.exercise2.model import F,jacobian,phi,grad_phi; print(F([0,-1]),jacobian([0,-1]),phi([0,-1]),grad_phi([0,-1]))"` |
| 2(b) | `python -m sc_cw1.exercise2.part_b_contours` |
| 2(c) | `python -m sc_cw1.exercise2.part_c_intersections` |
| 2(d) | `python -m sc_cw1.exercise2.part_d_gradient_descent` |
| 2(e) | `python -m sc_cw1.exercise2.part_e_momentum` |
| 2(f) | `python -m sc_cw1.exercise2.part_f_comparison` |

All numerical question modules accept `--output results`. Analytical parts 1(a)
and 2(a) define functions rather than experiments; their derivations still need
to be presented in the report. No complete coursework report is generated.

## Parameters and numerical conventions

| Experiment | Settings |
| --- | --- |
| 1(b), 1(c) | Newton with exact Hessian, objective Armijo; `tol=1e-6`, `maxit=200`; demonstration/search runs explicitly enable steepest-descent fallback |
| 1(c) | Unchanged 21×21 starts spanning [-2,2]²; clustering distance 1e-4; classification eigenvalue threshold 1e-7; separate exact rational global proof |
| 1(d) | Inverse BFGS, objective Armijo, B0=.01 I, `tol=1e-6`, `maxit=200`; invalid curvature fails explicitly |
| 1(e) | **Prescribed** x0=(-2,1), alpha=1 for both solvers, BFGS omega=.01; no line search/fallback/skipped update; supplementary `tol=1e-6`, `maxit=200` |
| 2(b) | 401×401 residual grid, supplementary local refinement; resolution checks at 201,401,801 |
| 2(c) | 2001 samples per zero curve; piecewise-linear graphical estimates; separate bisection refinement |
| 2(d), 2(f) | **Prescribed** x0=(0,-1), lambda=.05, tol=1e-6, maxit=200; 2(f) gamma=.1,.3,.7 |
| 2(e), 2(f) | Previous momentum d_-1=0 (assumption); printed recurrence with no gradient rescaling |

The Exercise 1 stopping settings, search grid, line-search constants and
safeguards are implementation choices where the brief gives no numerical
values. Armijo starts at 1, halves rejected steps, uses c1=1e-4, and allows 50
backtracks. All solvers use **strict `< tol`**. Exercise 1 checks the gradient
norm; Exercise 2 checks the residual norm, following Unit 10. A small step or
small residual gradient never substitutes for the Exercise 2 stopping statistic.

`newton(..., fallback="none")` is the library default; experiments 1(b/c)
explicitly opt into `fallback="steepest_descent"`. Histories record ordinary
Newton steps and fallback reasons separately. Fixed Newton is a stationarity
solver and may approach a saddle or maximum; classification is a separate check.
Fallbacks are disabled for 1(e).

`bfgs` implements the **expanded inverse** update printed in the brief, without
Hessians or linear solves. Nonpositive/unsafe curvature fails by default. An
explicit `curvature_action="skip"` option exists for additional experiments,
but none of the required runs uses it.

## Reading results and reviewing the implementation

- [CODE_MAP.md](CODE_MAP.md): every part, function, shared mathematical helper and output.
- [RESULTS.md](RESULTS.md): computed coordinates, convergence and trajectory evidence.
- [VALIDATION.md](VALIDATION.md): actual verification performed and its limits.
- [GLOBAL_MINIMUM_ANALYSIS.md](GLOBAL_MINIMUM_ANALYSIS.md): reduction, tail bounds, exact root count and unique global minimum.
- [REVIEW_REQUIRED.md](REVIEW_REQUIRED.md): classified conventions and remaining reporting limitations.
- [REPORT_SNIPPETS.md](REPORT_SNIPPETS.md): literal report-ready source excerpts, checked against the executed algorithms.
- [REVIEW1_CHANGES.md](REVIEW1_CHANGES.md): substantive modifications and baseline comparison.
- [REPORT_CODE_GUIDE.md](REPORT_CODE_GUIDE.md): source listings and outputs for the separate report.
- [SOURCE_AUDIT.md](SOURCE_AUDIT.md): source authority, equation checks and lecture support.
- [REVIEW_HANDOFF.md](REVIEW_HANDOFF.md): concise independent-review brief.
- [COMPLETION_REPORT.md](COMPLETION_REPORT.md): full status table, output inventory and reproduction commands.

Solver results contain `history`, `status`, `message`, `parameters`, `nit`,
`iterates`, `x` and `converged`. Histories include x0; `nit` counts completed
finite updates, excluding rejected trials. Each row's `alpha`/direction describes
the incoming step; row zero has alpha=0. Exit states are `converged`,
`max_iterations`, `numerical_failure`, and `line_search_failure`. Invalid API
controls raise `ValueError`; floating-point failures return a status and retain
the last finite iterate. Failed initial evaluations have empty histories.
Undefined summary metrics (e.g. an F residual in Exercise 1) are CSV `nan` and
JSON `null`, not evidence of a solver failure. All required runs are finite.

The source documents remain outside the project and are ignored if copied in.
Only authored code/documentation and generated results should go to GitHub.
Nothing is committed, pushed or submitted automatically.
