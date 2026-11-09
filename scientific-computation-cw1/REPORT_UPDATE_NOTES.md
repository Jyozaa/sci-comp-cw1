# CW1 report update notes

Reproduce these additions from `scientific-computation-cw1/` with
`python scripts/prepare_report_updates.py` using the existing environment.
The script calls the original solvers with their original parameters and checks
every returned history row against the existing CSVs before writing additions.
It does not overwrite any original figure, numerical result or report file.

## Exercise 2(f): separate momentum trajectories

New 600 DPI figure:
[results/figures/ex2_part_f_momentum_subplots.png](results/figures/ex2_part_f_momentum_subplots.png).
Paths here are relative to `scientific-computation-cw1/`.
All panels use the same phi contours (16 levels from 1e-5 to 10) and limits
derived from every trajectory point, with 8% padding:
x in [-0.063761, 0.860776], y in [-1.154302, 1.083083].
Black squares mark the common start (0,-1); black stars mark the final points.
Each title gives the actual update count. All runs converge using step .05,
residual tolerance 1e-6 and at most 200 updates, with fresh zero momentum.
The original overlaid figure remains available at
[results/figures/ex2_part_f_momentum_comparison.png](results/figures/ex2_part_f_momentum_comparison.png).
Use the new figure at approximately full text width; its fonts are 10–12 points
at the saved 7.2-inch figure width, with one shared legend.

## Exercise 2(d): selected ordinary-descent iterates

CSV: [results/tables/ex2_part_d_selected_iterations.csv](results/tables/ex2_part_d_selected_iterations.csv).
The residual column is ||F||₂; the gradient column is ||2 J_F.T F||₂.
The start is (0,-1), step .05, stop ||F||₂<1e-6, and maxit=200.
k counts accepted updates; k=0 is the initial point.

| k | x | y | residual_norm | gradient_norm |
| --- | --- | --- | --- | --- |
| 0 | 0 | -1 | 2.23606798 | 8.94427191 |
| 10 | 0.737819815 | -0.199908244 | 0.938824368 | 0.269290217 |
| 40 | 1.33706207 | -0.931246295 | 0.0430391517 | 0.0954719931 |
| 80 | 1.3768704 | -0.926163724 | 0.000479147517 | 0.000967956727 |
| 138 | 1.37733596 | -0.92607624 | 9.44831428e-07 | 1.9064809e-06 |

- From k=0 to k=10 the residual falls from 2.23607
  to 0.938824, while the gradient norm falls from
  8.94427 to 0.26929.
  Thus a relatively small gradient occurs while the residual is still about .94.
  On the update ending at k=10 the residual ratio is 0.997612783, a reduction
  of only 0.239%. Small residual-gradient norm alone does not
  establish a root; these data do not classify that region as a saddle.
- Near the final root, both norms decrease, but the absolute changes become
  small. The per-update residual ratios at k=40, 80 and 138 are respectively
  0.874690440, 0.897953023 and 0.898212035. The later ratios are
  closer to one, so fractional progress slows somewhat relative to k=40.
- The last-step factor remains 0.898212035, approximately .8982: about
  10.18% residual reduction on that update. This is an
  observed history ratio, not a convergence-order theorem or eigenvalue analysis.

## Exercise 1(e): selected gradient norms

CSV: [results/tables/ex1_part_e_selected_gradient_norms.csv](results/tables/ex1_part_e_selected_gradient_norms.csv).
Both columns give ||grad J(x_k)||₂. Settings: x0=(-2,1), fixed alpha=1,
Armijo disabled, BFGS B0=.01 I, gradient tolerance 1e-6, maxit=200.
Empty cells mean that the algorithm had already terminated; final values
are not repeated at later iteration numbers.

| k | Newton | BFGS |
| --- | --- | --- |
| 0 | 25.8655467 | 25.8655467 |
| 1 | 6.33904955 | 9.84977786 |
| 3 | 0.0654308687 | 3.43554427 |
| 5 | 5.88757448e-10 | 2.525529 |
| 8 |  | 0.19048808 |
| 12 |  | 1.67585217e-07 |

Newton takes 5 updates; BFGS takes 12.
Newton's norm falls from 0.0654309 at k=3 to
5.88757e-10 at its final update. BFGS makes slower
initial progress here: at k=3 its norm is 3.43554.
BFGS starts with a scalar inverse-Hessian approximation .01 I and adjusts it
using successive s/y pairs, whereas Newton uses the current Hessian. This
explains the different directions and is consistent with the early histories.
A Newton iteration evaluates the Hessian and solves H*delta=-g; BFGS uses
matrix-vector/outer products and its inverse update, without a Hessian solve.
Both evaluate the objective and gradient at new points. Counts alone do not
measure runtime, and this comparison concerns only the prescribed start,
scaling and fixed steps; it is not a general ranking or convergence theorem.

## Exercise 2(e): one coherent momentum excerpt

Source: `src/sc_cw1/exercise2/part_d_gradient_descent.py:gradient_descent`.
The following is one exact continuous function-body excerpt, dedented only.
The surrounding imports, function signature, parameter validation and final
return dictionary are omitted; all initialisation, numerical safeguards and
iteration recording within the excerpt are retained. NumPy is `np`;
`residual` is the existing F, `jac` its analytical Jacobian. Parameters are
x0=(0,-1), step_size=.05, tol=1e-6 and maxit=200. The momentum wrapper supplies
gamma; gamma=0 gives ordinary gradient descent in this very same loop.

```python
k = 0
x = np.array(x0, dtype=float)
d_prev = np.zeros(2)  # Momentum change 1: start with zero previous direction.
iterates = [x.copy()]
directions = [d_prev.copy()]
alphas = [0.0]
status = "max_iterations"
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    residual_value = residual(x)
    g = 2*jac(x).T @ residual_value
    values = [residual_value @ residual_value]
    residual_norm = np.linalg.norm(residual_value)
    residual_norms = [residual_norm]
    gradient_norms = [np.linalg.norm(g)]
    if not np.isfinite(values[-1]) or not np.isfinite(gradient_norms[-1]):
        status = "numerical_failure"
    while residual_norm >= tol and k < maxit and status != "numerical_failure":
        # Momentum change 2: gamma=0 gives ordinary gradient descent.
        d = gamma*d_prev + g
        x_new = x-step_size*d
        # Evaluate F and its analytical residual gradient at the new point.
        residual_new = residual(x_new)
        g_new = 2*jac(x_new).T @ residual_new
        residual_norm_new = np.linalg.norm(residual_new)
        grad_norm_new = np.linalg.norm(g_new)
        value_new = residual_new @ residual_new
        if (not np.isfinite(residual_norm_new) or not np.isfinite(grad_norm_new)
                or not np.isfinite(value_new)):
            status = "numerical_failure"
            break
        x, residual_value, g = x_new, residual_new, g_new
        residual_norm = residual_norm_new
        iterates.append(x.copy())
        values.append(value_new)
        residual_norms.append(residual_norm_new)
        gradient_norms.append(grad_norm_new)
        d_prev = d.copy()  # Momentum change 3: retain the accepted direction.
        directions.append(d.copy())
        alphas.append(step_size)
        k += 1
if status != "numerical_failure" and residual_norm < tol:
    status = "converged"
```

The return below this excerpt stores the accepted iterates/norms, status and
iterations=k. The three marked momentum changes initialise d_prev, add its
weighted value to g, and retain the accepted direction. The recurrence and
residual stopping test are exactly those executed in the coursework runs.

## Qualifications for the separate LaTeX revision

Insert the new figure, select the two compact evidence tables, and replace the
fragmented momentum listings with the excerpt as space permits. Tables retain
full precision in CSV; Markdown display rounds to nine significant digits.
The initial point is not an update. No new optimisation method, numerical
study, eigenvalue calculation or formal global-minimum proof is supplied here.
Existing limits on global-minimum justification still apply. No LaTeX, PDF,
References or Generative AI appendix was edited.
