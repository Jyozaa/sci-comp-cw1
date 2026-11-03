# Report source excerpts

These excerpts are copied directly from the executed source; they are not a
second implementation. The source paths and function names below identify
where to extract each listing. Select and format the relevant blocks within
the report's 15-page budget. Separate model and algorithm blocks can be used
as consecutive listings. No LaTeX report is generated here.

## Exercise 1(b): model, strict Newton and Armijo

Source: `src/sc_cw1/exercise1/model.py`, NumPy import and the complete
`objective`, `gradient`, `hessian` functions. Local `u,v` mean `x1,x2`.
These definitions supply the analytical derivatives used by the iteration.

```python
import numpy as np


def objective(x):
    """J=exp((u²+v²)/2)+sin²(2uv)+2(u-1/2)(v+1/2)."""
    u, v = np.asarray(x, dtype=float)
    return float(np.exp((u*u + v*v) / 2) + np.sin(2*u*v)**2 + 2*(u-.5)*(v+.5))


def gradient(x):
    """The first-order conditions set both components of this gradient to zero."""
    u, v = np.asarray(x, dtype=float)
    E, S = np.exp((u*u + v*v)/2), np.sin(4*u*v)
    # d(sin²(t))/dt=sin(2t), with t=2uv.
    return np.array([u*E + 2*v*S + 2*v + 1, v*E + 2*u*S + 2*u - 1])


def hessian(x):
    """Second derivatives of J; H12=H21 by symmetry."""
    u, v = np.asarray(x, dtype=float)
    E, C = np.exp((u*u + v*v)/2), np.cos(4*u*v)
    off = u*v*E + 2*np.sin(4*u*v) + 8*u*v*C + 2
    return np.array([[(1+u*u)*E+8*v*v*C, off], [off, (1+v*v)*E+8*u*u*C]])
```

Source: `src/sc_cw1/exercise1/part_b_newton.py`, complete `armijo`.
It tests sufficient decrease of the objective, starting at alpha=1 and halving
rejected steps. The constants and backtracking limit are retained conventions.

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

Source: the same file, complete `newton`. Its imported model names refer to
the functions above. The explicit counter records accepted updates; failures
leave the last accepted point. There is no replacement direction. Setting
`line_search=False` preserves the fixed unit step required in 1(e).

```python
def newton(x0, tol=1e-6, maxit=200, line_search=True,
           fun=objective, grad=gradient, hess=hessian):
    """Solve H*delta=-grad; stop at ||grad||₂<tol or after maxit updates.

    An unusable Newton direction fails the run; no other direction is used.
    Fixed comparison runs use line_search=False and exactly unit steps.
    """
    if tol <= 0 or maxit < 0:
        raise ValueError("tol must be positive and maxit nonnegative")
    k = 0
    x = np.array(x0, dtype=float)
    iterates = [x.copy()]
    alphas = [0.0]
    step_types = ["initial"]
    directions = [np.zeros(2)]
    status = "max_iterations"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value, g = fun(x), grad(x)
        grad_norm = np.linalg.norm(g)
        values, gradient_norms = [value], [grad_norm]
        if not np.all(np.isfinite(x)) or not np.isfinite(value) or not np.isfinite(grad_norm):
            status = "numerical_failure"
        while grad_norm >= tol and k < maxit and status != "numerical_failure":
            # Evaluate the Hessian.
            H = hess(x)
            if not np.all(np.isfinite(H)):
                status = "numerical_failure"
                break
            try:
                if np.linalg.cond(H) > 1e12:
                    raise np.linalg.LinAlgError("singular or poorly conditioned Hessian")
                # Calculate the Newton direction.
                delta = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                status = "numerical_failure"
                break
            if line_search and g @ delta >= 0:
                status = "line_search_failure"
                break
            # Choose an Armijo step, or the prescribed unit step in 1(e).
            alpha = armijo(fun, x, value, g, delta) if line_search else 1.0
            if alpha is None:
                status = "line_search_failure"
                break
            # Evaluate the new point before accepting the update.
            x_new = x+alpha*delta
            J_new, g_new = fun(x_new), grad(x_new)
            grad_norm_new = np.linalg.norm(g_new)
            if (not np.isfinite(J_new) or not np.isfinite(grad_norm_new)
                    or not np.all(np.isfinite(x_new)) or np.array_equal(x_new, x)):
                status = "numerical_failure"
                break
            x, value, g = x_new, J_new, g_new
            grad_norm = grad_norm_new
            iterates.append(x.copy())
            values.append(value)
            gradient_norms.append(grad_norm_new)
            alphas.append(alpha)
            step_types.append("newton")
            directions.append(delta.copy())
            k += 1
    if status != "numerical_failure" and grad_norm < tol:
        status = "converged"
    return dict(x=iterates[-1], iterates=np.array(iterates), values=np.array(values),
                gradient_norms=np.array(gradient_norms), alphas=np.array(alphas),
                step_kinds=step_types, directions=np.array(directions),
                iterations=k, status=status)
```

The final check recognises convergence on the last permitted update.
The condition-number limit and finite/stagnation checks prevent unusable
results from being reported as convergence. Complete functions are shown here;
only imports already supplied above and the command-line demonstration are
omitted. `step_kinds` is the existing result key; `step_types` is its local list.

## Exercise 1(d): inverse BFGS

Source: `src/sc_cw1/exercise1/part_d_bfgs.py`, complete
`inverse_bfgs_update` and `bfgs`. They use NumPy and the model/Armijo functions
above. The expanded inverse formula is unchanged. Invalid curvature fails
explicitly; no damping, skipped update or matrix reset is used.

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

```python
def bfgs(x0, omega=1.0, tol=1e-6, maxit=200, line_search=True,
         fun=objective, grad=gradient):
    """Use delta=-B*grad and stop at ||grad||₂<tol; invalid updates fail explicitly."""
    if omega <= 0 or tol <= 0 or maxit < 0:
        raise ValueError("omega/tol must be positive and maxit nonnegative")
    k = 0
    x = np.array(x0, dtype=float)
    B = omega*np.eye(2)
    iterates = [x.copy()]
    alphas = [0.0]
    directions = [np.zeros(2)]
    status = "max_iterations"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value, g = fun(x), grad(x)
        grad_norm = np.linalg.norm(g)
        values, gradient_norms = [value], [grad_norm]
        if not np.all(np.isfinite(x)) or not np.isfinite(value) or not np.isfinite(grad_norm):
            status = "numerical_failure"
        while grad_norm >= tol and k < maxit and status != "numerical_failure":
            # Calculate the direction using the inverse-Hessian approximation.
            delta = -B @ g
            alpha = armijo(fun, x, value, g, delta) if line_search else 1.0
            if alpha is None:
                status = "line_search_failure"
                break
            # Evaluate the trial step and its new gradient.
            s = alpha*delta
            x_new = x+s
            J_new, g_new = fun(x_new), grad(x_new)
            grad_norm_new = np.linalg.norm(g_new)
            if (not np.isfinite(J_new) or not np.isfinite(grad_norm_new)
                    or not np.all(np.isfinite(x_new)) or np.array_equal(x_new, x)):
                status = "numerical_failure"
                break
            # The gradient difference supplies the curvature information.
            y = g_new-g
            x, value, g = x_new, J_new, g_new
            grad_norm = grad_norm_new
            iterates.append(x.copy())
            values.append(value)
            gradient_norms.append(grad_norm_new)
            alphas.append(alpha)
            directions.append(delta.copy())
            k += 1
            if grad_norm_new < tol:
                break  # No further direction needs B after convergence.
            try:
                # Apply the prescribed inverse-BFGS update.
                B = inverse_bfgs_update(B, s, y)
            except ValueError:
                status = "numerical_failure"
                break
    if status != "numerical_failure" and grad_norm < tol:
        status = "converged"
    return dict(x=iterates[-1], iterates=np.array(iterates), values=np.array(values),
                gradient_norms=np.array(gradient_norms), alphas=np.array(alphas),
                directions=np.array(directions), step_kinds=["initial"]+["bfgs"]*(len(iterates)-1),
                iterations=k, status=status)
```

The listing includes B0=omega I, delta=-B g, s=alpha delta, y=g_new-g,
the update, stopping and counter. An accepted step is counted even if the
subsequent matrix update fails. After convergence no further matrix is needed.
In 1(e), x0=(-2,1), omega=.01 and `line_search=False`; alpha is exactly one.
No Hessian evaluation or linear solve occurs in BFGS.

## Exercise 2(e): residual gradient and momentum changes

Source: `src/sc_cw1/exercise2/model.py`, complete `F` and `jacobian`.
NumPy is imported as above. These analytical functions are passed as the
`residual` and `jac` arguments of the shared iteration.

```python
def F(x):
    """F1=(u-1)²+v²-1; F2=cos(2u)-v."""
    u, v = np.asarray(x, dtype=float)
    return np.array([(u-1)**2 + v**2 - 1, np.cos(2*u) - v])


def jacobian(x):
    """J_ij=∂F_i/∂x_j."""
    u, v = np.asarray(x, dtype=float)
    return np.array([[2*(u-1), 2*v], [-2*np.sin(2*u), -1.]])
```

Source: `src/sc_cw1/exercise2/part_d_gradient_descent.py`,
`gradient_descent`. Ordinary descent is the gamma=0 case of this actual loop.
The first momentum modification initialises fresh direction memory:

```python
k = 0
x = np.array(x0, dtype=float)
d_prev = np.zeros(2)  # Momentum change 1: start with zero previous direction.
```

After creating the history lists, the initial analytical residual gradient is
computed as follows (literal consecutive lines):

```python
residual_value = residual(x)
g = 2*jac(x).T @ residual_value
```

The omitted setup records phi=F.T F, `residual_norm=||F||₂` and the gradient
norm, and marks a nonfinite initial evaluation as failure. The following
complete loop is inside the function's existing `np.errstate` context. It
includes both remaining momentum modifications and all iteration recording:

```python
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
```

After leaving that context, the final stopping check is:

```python
if status != "numerical_failure" and residual_norm < tol:
    status = "converged"
```

These are function-body excerpts, not standalone programs. The omitted return
constructs the existing dictionary of arrays and `iterations=k`. The small
`momentum_descent` wrapper in `exercise2/part_e_momentum.py` checks 0<gamma<1
and calls this same loop; it creates no independent algorithm.

The recurrence is exactly d=gamma*d_prev+2J.T F and x_new=x-step_size*d.
With gamma=0 it is ordinary descent. The prescribed step is .05, start (0,-1),
tol=1e-6 and maxit=200; each gamma=.1,.3,.7 comparison starts with fresh memory.
Stopping is strictly ||F||₂<tol. The loop uses >= at equality and k<maxit,
so the initial point and unsuccessful attempts do not consume update counts.
