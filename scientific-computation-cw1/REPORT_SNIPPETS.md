# Report code excerpts

These are literal excerpts of the executed functions, rather than standalone
replacement algorithms. NumPy is imported as `np`. Include only the excerpts
needed for the three requested listings within the report's 15-page limit.
`CODE_MAP.md` identifies the complete functions and associated figures.

## Exercise 1(b): Newton and Armijo

Source: `src/sc_cw1/exercise1/part_b_newton.py`, `newton`.
Setup copies `x0`, evaluates `value=fun(x)` and the analytical `g=grad(x)`,
and records their values and norms in ordinary lists. `norms[-1]` is the
current gradient norm. The iteration allows at most `maxit` updates:

```python
for _ in range(maxit):
    if status == "numerical_failure" or norms[-1] < tol:
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
xn = x+alpha*delta
vn, gn = fun(xn), grad(xn)
nn = np.linalg.norm(gn)
if (not np.isfinite(vn) or not np.isfinite(nn)
        or not np.all(np.isfinite(xn)) or np.array_equal(xn, x)):
    status = "numerical_failure"
    break
x, value, g = xn, vn, gn
```

The omitted continuation appends the accepted point, objective, new norm `nn`,
alpha, direction and step kind `"newton"` to the history lists. After the loop
it checks the final norm again, so convergence on the last permitted update is recognised.
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
        return None
    alpha = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_backtracks+1):
            candidate = fun(x+alpha*delta)
            if np.isfinite(candidate) and candidate <= value+c1*alpha*slope:
                return alpha
            alpha *= 0.5
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
    if status == "numerical_failure" or norms[-1] < tol:
        break
    delta = -B @ g
    alpha = armijo(fun, x, value, g, delta) if line_search else 1.0
    if alpha is None:
        status = "line_search_failure"
        break
    s = alpha*delta
    xn = x+s
    vn, gn = fun(xn), grad(xn)
    nn = np.linalg.norm(gn)
    if (not np.isfinite(vn) or not np.isfinite(nn)
            or not np.all(np.isfinite(xn)) or np.array_equal(xn, x)):
        status = "numerical_failure"
        break
    y = gn-g
    x, value, g = xn, vn, gn
```

The omitted five lines append the accepted point, objective, norm, alpha and
direction to ordinary history lists. The continuation below avoids a needless
inverse update once the newly evaluated gradient meets tolerance:

```python
if nn < tol:
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
    By = B @ y
    updated = B + ((sy + y @ By) / sy**2) * np.outer(s, s)
    updated -= (np.outer(By, s) + np.outer(s, y @ B)) / sy
    if not np.all(np.isfinite(updated)):
        raise ValueError("nonfinite inverse-Hessian update")
    return updated
```

## Exercise 2(e): exact changes to ordinary descent

Sources: `src/sc_cw1/exercise2/part_d_gradient_descent.py`, `gradient_descent`;
`src/sc_cw1/exercise2/part_e_momentum.py`, `momentum_descent`.

The ordinary method uses `gamma=0`: its direction is `d=g=2*jac(x).T @ F(x)`
and its update is `xn=x-step_size*d`. The three marked changes implement
momentum in that same loop. First initialise fresh memory in every run:

```python
previous = np.zeros(2)  # Momentum change 1: choose the initial previous direction to be zero.
```

Setup computes `f=residual(x)` and `g=2*jac(x).T @ f` using the analytical
Jacobian. The iteration tests the residual norm, not the gradient norm:

```python
for _ in range(maxit):
    if status == "numerical_failure" or residual_norms[-1] < tol:
        break
    # Momentum change 2: gamma=0 gives the ordinary direction d=g.
    d = gamma*previous + g
    xn = x-step_size*d
    fn = residual(xn)
    gn = 2*jac(xn).T @ fn
    rn, nn, vn = np.linalg.norm(fn), np.linalg.norm(gn), fn @ fn
    if not np.isfinite(rn) or not np.isfinite(nn) or not np.isfinite(vn):
        status = "numerical_failure"
        break
    x, f, g = xn, fn, gn
```

After recording the accepted point, value, residual norm and gradient norm,
update the momentum state before the next iteration:

```python
previous = d.copy()  # Momentum change 3: remember d for the next update.
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
