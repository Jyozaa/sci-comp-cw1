# Newton robustness report notes

Reproduce only these additions from `scientific-computation-cw1/` with
`python scripts/compare_newton_robustness.py`. The original 1(b/c) experiment
remains strict by default; the original output files are preserved.

## Modification and scope

`newton(..., fallback=False)` is the unchanged strict default. With
`fallback=True` **and** Armijo enabled, the usual H*delta=-g solve is attempted.
An unusable Hessian/solve, nonfinite direction or non-descent direction selects
delta=-g. If backtracking fails for a usable Newton direction, Armijo is retried
once with -g. A negative-gradient search is never retried a second time.
Every accepted step must pass the original objective Armijo condition.

For nonzero g, g.T*(-g)=-||g||₂²<0, giving a descent direction. This is a
safeguard using taught steepest descent, not a modification of the Newton
equation. `line_search=False` always stays pure Newton, even if fallback=True.
The result schema is unchanged; accepted `step_kinds` distinguish `newton`
and `gradient_fallback`. Counts below refer to accepted steps, not attempts.

## Same-grid comparison

Settings: 21×21 uniformly spaced starts in [-2,2]²; ||grad J||₂<1e-6;
maxit=200; Armijo initially alpha=1, c1=1e-4, halving, up to 50 backtracks.

| method | starts | converged | failed | runs_with_fallback | accepted_fallback_steps |
| --- | --- | --- | --- | --- | --- |
| Strict Newton | 441 | 148 | 293 | 0 | 0 |
| Safeguarded Newton | 441 | 441 | 0 | 293 | 681 |

| method | mean_converged_updates | max_converged_updates |
| --- | --- | --- |
| Strict Newton | 6.01351351 | 11 |
| Safeguarded Newton | 8.21088435 | 21 |

The means and maxima include only converged runs. Their populations differ,
so they do not measure efficiency on identical sets of starts. Actual failure
causes are:

- Strict Newton: 293 non descent newton direction.
- Safeguarded Newton: no failures.

## Stationary-point distribution and checks

| method | point_id | classification | x1 | x2 | value | runs_in_cluster |
| --- | --- | --- | --- | --- | --- | --- |
| Strict Newton | 1 | minimum | -1.18530257 | 1.18530257 | -1.49914483 | 63 |
| Strict Newton | 2 | minimum | -0.659465157 | 0.659465157 | -0.559928824 | 81 |
| Strict Newton | 3 | saddle | -0.742325362 | 0.742325365 | -0.555735152 | 4 |
| Safeguarded Newton | 1 | minimum | -1.18530257 | 1.18530257 | -1.49914483 | 205 |
| Safeguarded Newton | 2 | minimum | -0.659465157 | 0.659465157 | -0.559928824 | 232 |
| Safeguarded Newton | 3 | saddle | -0.742325362 | 0.742325365 | -0.555735152 | 4 |

The existing distance grouping and analytical-gradient/Hessian classification
are used. Full derivative norms, Hessian determinants and H11 appear in the
stationary-point CSV. Only converged runs are grouped. Both variants identify
the same two minima and saddle to the stopping/grouping tolerances. P1 is the
lowest minimum found numerically, not a proven global minimum.
The 4 safeguarded saddle
outcomes remain possible because the stopping test recognises stationarity,
which does not imply a minimum.

## Illustrative start (0,0)

At this point, J=0.5, g=[1.0, -1.0] and H=[[1.0, 2.0], [2.0, 1.0]].
The computed Newton direction is [1.0, -1.0], with g.T*delta=2:
it fails the descent test, so strict Newton terminates with
`line_search_failure` after 0 accepted updates.

The safeguarded run selects [-1.0, 1.0], whose
gradient-direction product is -2. Armijo accepts
alpha=1, reaching [-1.0, 1.0].
It finishes `converged` after 5 updates,
with 3 Newton and
2 fallback steps.
The final point is (-1.18530259, 1.18530259), with
J=-1.4991448256 and ||grad J||₂=6.47895392e-07.

| method | k | x1 | x2 | value | gradient_norm | alpha | step_kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Strict Newton | 0 | 0 | 0 | 0.5 | 1.41421356 | 0 | initial |
| Safeguarded Newton | 0 | 0 | 0 | 0.5 | 1.41421356 | 0 | initial |
| Safeguarded Newton | 1 | -1 | 1 | -0.954896361 | 2.53897036 | 1 | gradient_fallback |
| Safeguarded Newton | 2 | -1.2244154 | 1.2244154 | -1.44877454 | 1.8983907 | 0.125 | gradient_fallback |
| Safeguarded Newton | 3 | -1.18941238 | 1.18941238 | -1.49863265 | 0.177160771 | 1 | newton |
| Safeguarded Newton | 4 | -1.1853654 | 1.1853654 | -1.49914471 | 0.00266697394 | 1 | newton |
| Safeguarded Newton | 5 | -1.18530259 | 1.18530259 | -1.49914483 | 6.47895392e-07 | 1 | newton |

The origin history includes every accepted direction in CSV. An Armijo decrease
in J does not require the gradient norm to decrease on every step.

## Exact direction-selection source excerpt

Source: `src/sc_cw1/exercise1/part_b_newton.py:newton`. This is one continuous
excerpt from inside the main while loop, dedented only. The surrounding code
has evaluated g, checked the gradient tolerance and initial finiteness;
trial objective/gradient evaluation, finite-trial
checks and history recording follow it. Accepted updates append `step_type`
to `step_types` (returned as `step_kinds`) and only then increment k.

```python
H = hess(x)
step_type = "newton"
try:
    if not np.all(np.isfinite(H)) or np.linalg.cond(H) > 1e12:
        raise np.linalg.LinAlgError("singular or poorly conditioned Hessian")
    # Calculate the Newton direction.
    delta = np.linalg.solve(H, -g)
except np.linalg.LinAlgError:
    if not (fallback and line_search):
        status = "numerical_failure"
        break
    delta, step_type = -g, "gradient_fallback"
if fallback and line_search and (not np.all(np.isfinite(delta)) or not (g @ delta < 0)):
    # Negative gradient gives g.T*delta=-||g||² when Newton is unsuitable.
    delta, step_type = -g, "gradient_fallback"
if line_search and g @ delta >= 0:
    status = "line_search_failure"
    break
# Choose an Armijo step, or the prescribed unit step in 1(e).
alpha = armijo(fun, x, value, g, delta) if line_search else 1.0
if alpha is None and fallback and line_search and step_type == "newton":
    # Retry once with -g only if it has not already been tried.
    delta, step_type = -g, "gradient_fallback"
    alpha = armijo(fun, x, value, g, delta)
if alpha is None:
    status = "line_search_failure"
    break
# Evaluate the new point before accepting the update.
x_new = x+alpha*delta
```

## New output locations

All paths are relative to `scientific-computation-cw1/`:

- [Summary](results/tables/ex1_newton_robustness_summary.csv)
- [Stationary points and analytical checks](results/tables/ex1_newton_robustness_stationary_points.csv)
- [All 882 starting-point outcomes and accepted step counts](results/tables/ex1_newton_robustness_runs.csv)
- [Complete origin histories](results/tables/ex1_newton_origin_history.csv)
- [Sampled starting-point comparison](results/figures/ex1_newton_robustness_basins.png)

The figure plots **initial guesses**, coloured by the stationary point reached;
unsuccessful strict starts are grey crosses. Both panels have identical scales
and original-J contours. It depicts sampled destinations, not analytically
established basin boundaries. Original figures are not overwritten.

## Benefits, limitations and later report wording

The safeguard improves convergence coverage on this grid while explicitly
recording when another direction is used. These runs are safeguarded Newton,
not pure Newton on every step. Finite tests do not guarantee convergence for
arbitrary starts or establish global optimality. Numerical failure and search
exhaustion remain possible; failures are never classified as minima.
The existing one-variable/global-minimum discussion is unchanged.

For the separate 1(b/c) revision, retain the strict results as the baseline,
describe the safeguard as optional, include accepted fallback counts, and
distinguish objective descent from convergence to a minimum. Keep 1(e)'s
fixed-step Newton/BFGS comparison separate and unchanged. No LaTeX, PDF,
References or Generative AI appendix is edited by this script.
