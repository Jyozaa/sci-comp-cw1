# Report-ready code excerpts after Review 1

These are **literal, contiguous excerpts** of the algorithms actually executed,
not replacement solvers. The tests check each Python block against its source
file. Function setup/parameter validation is omitted where stated; excerpts
that start inside a function are not standalone runnable programs. Use the
smallest relevant excerpts within the report's 15-page limit, and cite the
full function for omitted bookkeeping. No complete coursework report is made.

## 1(b): Newton iteration and objective Armijo

`newton` receives `fun`, analytical `grad`, and exact or numerical `hess`.
Before the excerpt it validates controls, copies x0, constructs `result`, checks
finiteness, evaluates `f,g,gn=checked_evaluation(fun,grad,x)` and records x0.
The following is the **entire loop and final stopping check**, including the
condition limit, non-descent fallback, failure statuses and history records.
`fallback='steepest_descent'` is explicitly enabled in 1(b/c) experiments;
the library default is `'none'`. The `step_kind` field distinguishes every
fallback from an ordinary Newton step. In 1(e), `step_mode='fixed', alpha=1,
fallback='none'` bypasses line search and never replaces the Newton direction.

Source: `src/sc_cw1/exercise1/part_b_newton.py` — `newton`, lines 41–85.

```python
for _ in range(maxit):
    if gn < tol:
        return result.stop("converged", "gradient norm < tolerance")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            H = np.asarray(finite(hess(x)), dtype=float)
            if H.shape != (2, 2):
                raise ValueError("Hessian must have shape (2,2)")
            condition = float(np.linalg.cond(H))
            kind = "newton"
            if not np.isfinite(condition) or condition > condition_limit:
                if fallback == "none":
                    return result.stop("numerical_failure", "singular or ill-conditioned Hessian")
                p, kind = -g, "fallback_condition"
            else:
                try:
                    p = finite(np.linalg.solve(H, -g))
                except np.linalg.LinAlgError:
                    if fallback == "none":
                        raise
                    p, kind = -g, "fallback_singular"
            slope = float(finite(g @ p))
            if step_mode == "armijo" and slope >= 0:
                if fallback == "none":
                    return result.stop("line_search_failure", "Newton direction is not descent")
                p, kind = -g, "fallback_non_descent"
                slope = float(g @ p)
        a = alpha if step_mode == "fixed" else armijo(
            fun, x, f, g, p, initial_alpha=alpha, c1=c1,
            shrink=shrink, max_backtracks=max_backtracks)
        if a is None:
            return result.stop("line_search_failure", "Armijo backtracking exhausted")
        with np.errstate(over="raise", invalid="raise"):
            xn = finite(x + a*p)
        fn, gnew, gnnew = checked_evaluation(fun, grad, xn)
        if np.array_equal(xn, x):
            return result.stop("numerical_failure", "floating-point stagnation above tolerance")
    except NUMERICAL_ERRORS as exc:
        return result.stop("numerical_failure", str(exc))
    result.append(xn, fn, gnnew, a, step_kind=kind, direction1=float(p[0]),
                  direction2=float(p[1]), directional_derivative=slope, hessian_condition=condition)
    x, f, g, gn = xn, fn, gnew, gnnew
if gn < tol:
    return result.stop("converged", "gradient norm < tolerance at final allowed update")
return result
```

`armijo` first validates c1, shrink, initial_alpha and the backtracking budget.
Its remaining code below rejects non-descent directions, tests the **objective**
sufficient-decrease inequality, halves only rejected steps, and rejects
nonfinite trial values. No step is accepted after search exhaustion. This is
the stated adaptation of Unit 9's residual-based line-search material.

Source: `src/sc_cw1/common/utilities.py` — `armijo`, lines 124–140.

```python
with np.errstate(over="raise", invalid="raise"):
    slope = float(finite(gradient @ direction))
if slope >= 0:
    return None
alpha = initial_alpha
for _ in range(max_backtracks + 1):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            trial = finite(x + alpha * direction)
            candidate = float(finite(fun(trial)))
            bound = float(finite(value + c1 * alpha * slope))
        if candidate <= bound:
            return alpha
    except NUMERICAL_ERRORS:
        pass
    alpha *= shrink
return None
```

Helpers omitted from these excerpts: `vector` validates shape;
`finite` rejects NaN/Inf; `checked_evaluation` returns f, analytical g and ||g||₂
under floating-point checks; `NUMERICAL_ERRORS` names the handled numerical
exceptions. `SolverResult.append/stop` only records values/status, not a hidden
mathematical update. Armijo defaults are c1=1e-4, shrink=.5, initial alpha=1,
50 backtracks. Stop strictly at ||g||₂<tol; maxit counts updates, not x0.

## 1(d): expanded inverse-Hessian BFGS

The validated initial scaling is positive and initialises the inverse-Hessian
approximation. The normal BFGS iteration imports no Hessian and solves no
linear system.

Source: `src/sc_cw1/exercise1/part_d_bfgs.py` — `bfgs`, line 57.

```python
B = omega * np.eye(2)
```

After `result` is constructed, the following contiguous excerpt includes the
initial-state check, entire loop, gradient difference, prescribed step/update
sequence and statuses. With `step_mode='fixed'`, alpha is used literally;
1(e) sets alpha=1 and omega=.01. Every delivered run uses
`curvature_action='fail'`. The opt-in `'skip'` branch below is real library
functionality but is not used by the coursework experiments.

Source: `src/sc_cw1/exercise1/part_d_bfgs.py` — `bfgs`, lines 62–106.

```python
try:
    f, g, gn = checked_evaluation(fun, grad, x)
except NUMERICAL_ERRORS as exc:
    return result.stop("numerical_failure", f"initial evaluation: {exc}")
result.append(x, f, gn, step_kind="initial", update_action="initial")
if gn < tol:
    return result.stop("converged", "initial gradient norm < tolerance")
for _ in range(maxit):
    if gn < tol:
        return result.stop("converged", "gradient norm < tolerance")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            p = finite(-B @ g)
            slope = float(finite(g @ p))
        a = alpha if step_mode == "fixed" else armijo(
            fun, x, f, g, p, initial_alpha=alpha, c1=c1,
            shrink=shrink, max_backtracks=max_backtracks)
        if a is None:
            return result.stop("line_search_failure", "non-descent direction or Armijo exhausted")
        with np.errstate(over="raise", invalid="raise"):
            s = finite(a*p)
            xn = finite(x + s)
        fn, gnew, gnnew = checked_evaluation(fun, grad, xn)
        if np.array_equal(xn, x):
            return result.stop("numerical_failure", "floating-point stagnation above tolerance")
        with np.errstate(over="raise", invalid="raise"):
            y = finite(gnew - g)
            sy = float(finite(s @ y))
    except NUMERICAL_ERRORS as exc:
        return result.stop("numerical_failure", str(exc))
    result.append(xn, fn, gnnew, a, step_kind="bfgs", direction1=float(p[0]),
        direction2=float(p[1]), directional_derivative=slope, curvature=sy,
        update_action="not_needed_converged" if gnnew < tol else "pending")
    x, f, g, gn = xn, fn, gnew, gnnew
    if gn < tol:
        return result.stop("converged", "gradient norm < tolerance")
    try:
        B = inverse_bfgs_update(B, s, y, curvature_rtol)
        result.history[-1]["update_action"] = "inverse_bfgs"
    except NUMERICAL_ERRORS as exc:
        result.history[-1]["update_action"] = "failed_curvature"
        if curvature_action == "fail":
            return result.stop("numerical_failure", str(exc))
        result.history[-1]["update_action"] = "skipped_curvature"
return result
```

`inverse_bfgs_update` first validates curvature_rtol, vector dimensions and the
2×2 B shape. The exact remaining body below contains the printed expanded
**inverse** update. In particular both subtraction products and the squared
curvature denominator must remain in a report listing. The check is relative
to ||s||||y||; it does not edit s, y or B to manufacture positive curvature.

Source: `src/sc_cw1/exercise1/part_d_bfgs.py` — `inverse_bfgs_update`, lines 25–35.

```python
with np.errstate(over="raise", invalid="raise", divide="raise"):
    finite(B)
    sy = float(finite(s @ y))
    scale = float(finite(np.linalg.norm(s) * np.linalg.norm(y)))
    if sy <= 0 or sy <= curvature_rtol * scale:
        raise FloatingPointError("nonpositive or numerically unsafe BFGS curvature")
    By = B @ y
    # Outer products are essential: no elementwise stand-in for matrix products.
    updated = B + ((sy + y @ By) / sy**2) * np.outer(s, s)
    updated -= (np.outer(By, s) + np.outer(s, y @ B)) / sy
return finite(updated)
```

Here the helpers have the same finite-evaluation/recording roles as in 1(b).
The update is unnecessary after the newly evaluated gradient already meets
tolerance, because no further search direction will be used. The normal
iteration computes p=-Bg, s=alpha*p and y=g_new-g; `By=B@y` and outer products
implement matrix multiplication, not elementwise substitutes. Failure/skip
behavior is explicit; no Hessian reset, damping or altered BFGS formula occurs.

## 2(e): ordinary descent and exact momentum changes

Ordinary gradient descent is the gamma=0 case of the shared loop. Its original
mathematical update is x_(k+1)=x_k-lambda*(2J(x_k).T F(x_k)). The actual wrapper
below forces gamma=0 and uses the analytical residual/Jacobian by default.

Source: `src/sc_cw1/exercise2/part_d_gradient_descent.py` — `gradient_descent`, lines 77–78.

```python
return fixed_descent(x0, step_size=step_size, tol=tol, maxit=maxit,
                     residual=residual, jac=jac, gamma=0.)
```

The gradient supplied to that loop is evaluated at the **current** x_k, with
these actual residual-state lines (the extra values support recording and the
residual-norm stopping test):

Source: `src/sc_cw1/exercise2/part_d_gradient_descent.py` — `residual_state`, lines 13–22.

```python
with np.errstate(over="raise", invalid="raise", divide="raise"):
    finite(x)
    f = vector(finite(residual(x)))
    J = np.asarray(finite(jac(x)), dtype=float)
    if J.shape != (2, 2):
        raise ValueError("Jacobian must have shape (2,2)")
    g = finite(2*J.T @ f)
    value = float(finite(f @ f))
    rn, gn = float(finite(np.linalg.norm(f))), float(finite(np.linalg.norm(g)))
return f, value, g, rn, gn
```

**Change 1:** retain a previous direction. Zero d_-1 is an explicitly stated
convention, and each run gets fresh state; callers may supply another initial
direction for separately labelled experiments.

Source: `src/sc_cw1/exercise2/part_d_gradient_descent.py` — `fixed_descent`, line 40.

```python
previous = np.zeros(2) if initial_direction is None else vector(initial_direction)
```

After validating and recording the initial residual state, this is the entire
shared loop and final stopping check. **Change 2** adds gamma*previous to the
current gradient; **change 3** saves d_k for the next iteration. With gamma=0
it is ordinary descent. `momentum_descent` in `exercise2/part_e_momentum.py`
requires 0<gamma<1 and delegates to this same loop without other changes.

Source: `src/sc_cw1/exercise2/part_d_gradient_descent.py` — `fixed_descent`, lines 51–71.

```python
for _ in range(maxit):
    if rn < tol:
        return result.stop("converged", "residual norm < tolerance")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # MOMENTUM CHANGE 2: accumulate previous gradient information.
            # gamma=0 gives d_k=grad_phi, the original descent direction.
            direction = finite(gamma*previous + g)
            xn = finite(x - step_size*direction)
        fn, vn, gnew, rnnew, gnnew = residual_state(xn, residual, jac)
    except NUMERICAL_ERRORS as exc:
        return result.stop("numerical_failure", str(exc))
    result.append(xn, vn, gnnew, step_size, residual_norm=rnnew,
                  f1=float(fn[0]), f2=float(fn[1]), direction1=float(direction[0]),
                  direction2=float(direction[1]), step_kind="momentum" if gamma else "gradient_descent")
    # MOMENTUM CHANGE 3: retain d_k for the next iteration (no reset).
    previous = direction.copy()
    x, f, value, g, rn, gn = xn, fn, vn, gnew, rnnew, gnnew
if rn < tol:
    return result.stop("converged", "residual norm < tolerance at final allowed update")
return result
```

This is exactly d_k=gamma*d_(k-1)+2J.T F, x_(k+1)=x_k-lambda*d_k.
There is no (1-gamma) factor, look-ahead gradient, adaptive lambda, memory reset
or extra 201st update. The `residual_state` helper above recomputes F, J and g
at x_(k+1), which becomes the current state for the next iteration. The chosen
stopping statistic is ||F||₂<tol, supported by Unit 10; it is not ||grad_phi||.

For 2(d/f), lambda=.05, x0=(0,-1), tol=1e-6, maxit=200; 2(f) independently
uses gamma=.1,.3,.7. A convergence exit can occur earlier than 200 updates;
otherwise exactly 200 updates are allowed. Recording does not change the
recurrence. REPORT_CODE_GUIDE.md maps the associated figures/tables; the
classification of every convention is in REVIEW_REQUIRED.md.
