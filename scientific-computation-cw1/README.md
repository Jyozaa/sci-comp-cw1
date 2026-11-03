# Scientific Computation Coursework 1

Question-by-question Python code for COMP5930M CW1. The implementation follows
the original brief and the ordinary function/NumPy/loop style of the supplied
notebooks. It uses NumPy, Matplotlib and pytest, with no external solvers.

## Install and run

Use Python 3.12 or newer (tested here with 3.13.5):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install . --no-deps
python -m pytest -q
python scripts/run_all.py
```

The last command uses the current source tree and regenerates the six required
300 DPI PNGs in `results/figures/`, eleven CSVs in `results/tables/`, and
RESULTS.md. It takes seconds on a normal laptop. No source PDF, ZIP, internet
service, GPU or random seed is needed to run it. Source materials stay outside
the repository. Use `--output /tmp/cw1/results` for another destination;
RESULTS.md is written in that directory's parent.

After source edits, reinstall with `python -m pip install . --no-deps --force-reinstall`
before using the independent `python -m` commands, which import the installed
package. pytest and `scripts/run_all.py` use `src/` directly.

## Find or run a question

| Part | Location / independent command |
| --- | --- |
| 1(a) | `src/sc_cw1/exercise1/model.py`: objective, gradient, Hessian |
| 1(b) | `python -m sc_cw1.exercise1.part_b_newton` (prints one demonstration; 1(c) runs multiple starts) |
| 1(c) | `python -m sc_cw1.exercise1.part_c_minima` |
| 1(d) | `python -m sc_cw1.exercise1.part_d_bfgs` (prints one demonstration; 1(e) compares convergence) |
| 1(e) | `python -m sc_cw1.exercise1.part_e_comparison` |
| 2(a) | `src/sc_cw1/exercise2/model.py`: residual, Jacobian and squared residual |
| 2(b) | `python -m sc_cw1.exercise2.part_b_contours` |
| 2(c) | `python -m sc_cw1.exercise2.part_c_intersections` |
| 2(d) | `python -m sc_cw1.exercise2.part_d_gradient_descent` |
| 2(e) | `python -m sc_cw1.exercise2.part_e_momentum` (prints one example; 2(f) supplies all three runs) |
| 2(f) | `python -m sc_cw1.exercise2.part_f_comparison` |

All numerical modules accept `--output results`. The analytical parts still
need their written derivations in the separately prepared report.

## Settings and simple conventions

- Exercise 1: analytical gradient/Hessian; gradient norm strictly below 1e-6;
  at most 200 updates. These stopping numbers are implementation choices.
  The multistart grid is 21×21 over [-2,2]²; points within 1e-4 are merged.
- Newton/Armijo starts at alpha=1 and halves rejected steps (c1=1e-4, up to
  50 halvings). 1(b/c) use only the Newton direction. A singular/poorly
  conditioned Hessian (limit 1e12) gives `numerical_failure`; a non-descent
  direction or exhausted Armijo search gives `line_search_failure`. No gradient
  descent is substituted. All 441 starts are attempted; only converged runs
  are classified (148 converged, 293 non-descent failures in the current grid).
- **1(e):** prescribed x0=(-2,1), alpha=1 and BFGS omega=.01. Line search
  is disabled. Invalid BFGS curvature is reported as failure,
  never repaired by editing the printed inverse update or resetting B.
- **2(d/f):** prescribed x0=(0,-1), step=.05, tol=1e-6, maxit=200; 2(f) uses
  gamma=.1,.3,.7. Following Unit 10, stopping uses ||F||₂, not its square or the
  residual gradient. The stored momentum direction `d_prev` starts at zero.
- Histories include x0; iteration counts exclude it. At most 200 updates are
  permitted; convergence can occur earlier. Numerical failures and exhausted
  searches have explicit statuses and retain the last accepted point.
  Each solver uses an explicit `k` counter in a `while` loop; only accepted
  updates increment it. Momentum shares the gradient-descent loop.

Solvers return ordinary dictionaries containing NumPy arrays (`iterates`,
`values`, `gradient_norms`, `alphas`, `directions`) and `status`, `iterations`,
`x`. Exercise 2 also returns `residual_norms`. There is no result class.

[CODE_MAP.md](CODE_MAP.md) maps every calculation/output;
[RESULTS.md](RESULTS.md) gives the computed findings;
[REPORT_SNIPPETS.md](REPORT_SNIPPETS.md) gives the three requested code listings;
[REVIEW_REQUIRED.md](REVIEW_REQUIRED.md) records limitations and manual checks;
[AUDIT_CHANGELOG.md](AUDIT_CHANGELOG.md) contains the source-based checklist,
change/deletion inventory and actual verification. No final PDF report,
commit or push is generated automatically.
