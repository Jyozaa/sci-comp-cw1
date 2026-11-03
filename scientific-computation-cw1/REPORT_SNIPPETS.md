# Report code excerpts

These are literal excerpts of the executed functions, rather than standalone
replacement algorithms. NumPy is imported as `np`. Include only the excerpts
needed for the three requested listings within the report's 15-page limit.
`CODE_MAP.md` identifies the complete functions and associated figures.

## Exercise 1(b): Newton and Armijo

Source: `src/sc_cw1/exercise1/part_b_newton.py`, `newton`.
Setup copies `x0`, evaluates `value=fun(x)` and the analytical `g=grad(x)`,
and records their values and norms in ordinary lists. `grad_norm` is the
current gradient norm; `gradient_norms` stores its history. The iteration allows
at most `maxit` updates:

```python
for _ in range(maxit):
    if status == "numerical_failure" or grad_norm < tol:
        break
```

The following contiguous block performs the strict Newton solve, step selection
and trial evaluation. An unusable or non-descent Newton direction fails the run;
no other direction is substituted. This is the same loop used in 1(c).

```python
H = hess(x)
if not np.all(np.isfinite(H)):
    status = "numerical_failure"
    break
try:
    if np.linalg.cond(H) > 1e12:
        raise np.linalg.LinAlgError("singular or poorly conditioned Hessian")
    # Compute the Newton direction by solving H*delta=-grad.
    delta = np.linalg.solve(H, -g)
except np.linalg.LinAlgError:
    status = "numerical_failure"
    break
if line_search and g @ delta >= 0:
    status = "line_search_failure"
    break
alpha = armijo(fun, x, value, g, delta) if line_search else 1.0
if alpha is None:
    status = "line_search_failure"
    break
x_new = x+alpha*delta
J_new, g_new = fun(x_new), grad(x_new)
grad_norm_new = np.linalg.norm(g_new)
if (not np.isfinite(J_new) or not np.isfinite(grad_norm_new)
        or not np.all(np.isfinite(x_new)) or np.array_equal(x_new, x)):
    status = "numerical_failure"
    break
x, value, g = x_new, J_new, g_new
grad_norm = grad_norm_new
```

The omitted continuation appends the accepted point, objective,
`grad_norm_new`, alpha, direction and step kind `"newton"` to the history lists.
After the loop it checks the final norm again, so convergence on the last
permitted update is recognised.
A singular/poorly conditioned Hessian, exhausted line search, nonfinite trial
or unchanged floating-point iterate is reported as a failure rather than
convergence. The condition threshold is the stated convention `1e12`.

Source: the same file, `armijo` (entire function). Step-halving tests the
objective sufficient-decrease inequality; non-descent directions are rejected.

```python
def armijo(fun, x, value, g, delta, c1=1e-4, max_backtracks=50):
    """Find alpha with J(x+alpha*delta)<=J(x)+c1*alpha*g.T*delta."""
    slope = g @ delta
    if not np.isfinite(slope) or slope >= 0:
        return None  # Armijo requires a descent direction.
    alpha = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_backtracks+1):
            candidate = fun(x+alpha*delta)
            if np.isfinite(candidate) and candidate <= value+c1*alpha*slope:
                return alpha
            alpha *= 0.5  # Reduce the step until sufficient decrease holds.
    return None  # Do not accept a step after a failed search.
```

The brief requests Armijo; `c1=1e-4`, halving and 50 backtracks are implementation
conventions. In 1(e), `line_search=False` uses exactly alpha=1.

## Exercise 1(d): inverse-Hessian BFGS

Source: `src/sc_cw1/exercise1/part_d_bfgs.py`, `bfgs`.
After copying `x0`, initialise the inverse approximation as prescribed:

```python
B = omega*np.eye(2)
```

Setup evaluates the objective and gradient and records the initial norm.
This contiguous iteration excerpt shows the stop test, direction, step,
gradient difference and acceptance checks:

```python
for _ in range(maxit):
    if status == "numerical_failure" or grad_norm < tol:
        break
    delta = -B @ g
    alpha = armijo(fun, x, value, g, delta) if line_search else 1.0
    if alpha is None:
        status = "line_search_failure"
        break
    s = alpha*delta
    x_new = x+s
    J_new, g_new = fun(x_new), grad(x_new)
    grad_norm_new = np.linalg.norm(g_new)
    if (not np.isfinite(J_new) or not np.isfinite(grad_norm_new)
            or not np.all(np.isfinite(x_new)) or np.array_equal(x_new, x)):
        status = "numerical_failure"
        break
    y = g_new-g
    x, value, g = x_new, J_new, g_new
    grad_norm = grad_norm_new
```

The omitted five lines append the accepted point, objective, norm, alpha and
direction to ordinary history lists. The continuation below avoids a needless
inverse update once the newly evaluated gradient meets tolerance:

```python
if grad_norm_new < tol:
    break  # No further direction needs B after convergence.
try:
    B = inverse_bfgs_update(B, s, y)
except ValueError:
    status = "numerical_failure"
    break
```

The full function checks the final norm after the loop and returns the
histories and status. No Hessian is evaluated and no linear system is solved.
`line_search=False` bypasses Armijo in 1(e); omega=.01 there.

Source: the same file, `inverse_bfgs_update` (entire function).
The expanded inverse formula is unchanged from the brief. An unsafe denominator
causes explicit failure: there is no skipped update, damping or matrix reset.

```python
def inverse_bfgs_update(B, s, y):
    """Update the inverse approximation using the expanded formula in the brief."""
    sy = s @ y
    if not np.isfinite(sy) or sy <= 0 or sy <= 1e-12*np.linalg.norm(s)*np.linalg.norm(y):
        raise ValueError("nonpositive or unsafe BFGS curvature")
    B_y = B @ y
    B_new = B + ((sy + y @ B_y) / sy**2) * np.outer(s, s)
    B_new -= (np.outer(B_y, s) + np.outer(s, y @ B)) / sy
    if not np.all(np.isfinite(B_new)):
        raise ValueError("nonfinite inverse-Hessian update")
    return B_new
```

## Exercise 2(e): exact changes to ordinary descent

Sources: `src/sc_cw1/exercise2/part_d_gradient_descent.py`, `gradient_descent`;
`src/sc_cw1/exercise2/part_e_momentum.py`, `momentum_descent`.

The ordinary method uses `gamma=0`: its direction is `d=g=2*jac(x).T @ F(x)`
and its update is `x_new=x-step_size*d`. The three marked changes implement
momentum in that same loop. First initialise fresh memory in every run:

```python
d_prev = np.zeros(2)  # Momentum change 1: start with zero previous direction.
```

Setup computes `residual_value=residual(x)` and
`g=2*jac(x).T @ residual_value` using the analytical Jacobian. The iteration
tests the residual norm, not the gradient norm:

```python
for _ in range(maxit):
    if status == "numerical_failure" or residual_norms[-1] < tol:
        break
    # Momentum change 2: gamma=0 gives ordinary gradient descent.
    d = gamma*d_prev + g
    x_new = x-step_size*d
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
```

After recording the accepted point, value, residual norm and gradient norm,
update the momentum state before the next iteration:

```python
d_prev = d.copy()  # Momentum change 3: retain the accepted direction.
```

The remaining lines record `d` and the fixed step. The full function checks
`residual_norms[-1] < tol` again after the loop and returns a dictionary.
The small `momentum_descent` wrapper requires `0<gamma<1` and calls this
function with the requested gamma. There is no alternative recurrence or
custom initial memory. The zero previous direction is an explicit convention.

The executed settings are `step_size=.05`, `x0=(0,-1)`, `tol=1e-6`,
`maxit=200`. Each 2(f) run starts independently with gamma=.1, .3 or .7.
`range(maxit)` permits exactly 200 updates, with the initial point at k=0;
convergence can stop earlier. Only `||F||₂<tol` declares a root converged.
