"""Algorithm tests on independent quadratics, line search and failure paths."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sc_cw1.common.utilities import armijo, finite_difference_jacobian
from sc_cw1.exercise1.part_b_newton import newton
from sc_cw1.exercise1.part_d_bfgs import bfgs, inverse_bfgs_update

A = np.array([[4., 1.], [1., 3.]])
TARGET = np.array([1., -2.])


def quadratic(x):
    z = x-TARGET
    return .5*z@A@z


def quad_gradient(x):
    return A@(x-TARGET)


def test_newton_exact_quadratic_step():
    r = newton((7, 5), fun=quadratic, grad=quad_gradient, hess=lambda x: A, tol=1e-12)
    assert r.converged and r.nit == 1
    assert_allclose(r.x, TARGET, atol=1e-14)
    assert r.history[-1]["alpha"] == 1


@pytest.mark.parametrize("mode", ["fixed", "armijo"])
def test_bfgs_on_known_quadratic(mode):
    r = bfgs((7, 5), fun=quadratic, grad=quad_gradient, omega=.1,
             step_mode=mode, tol=1e-9)
    assert r.converged, r.message
    assert_allclose(r.x, TARGET, atol=1e-9)
    assert np.linalg.norm(quad_gradient(r.x)) < 1e-9
    if mode == "armijo":
        for before, after in zip(r.history, r.history[1:]):
            assert after["directional_derivative"] < 0
            assert after["value"] <= before["value"] + 1e-4*after["alpha"]*after["directional_derivative"]


def test_printed_inverse_bfgs_update_matches_compact_form_and_secant_equation():
    B = np.array([[.7, .1], [.1, .5]])
    s = np.array([.3, -.2])
    y = A@s
    result = inverse_bfgs_update(B, s, y)
    rho = 1/(s@y)
    V = np.eye(2)-rho*np.outer(s, y)
    compact = V@B@V.T + rho*np.outer(s, s)
    assert_allclose(result, compact, atol=1e-15)
    assert_allclose(result@y, s, atol=1e-15)
    assert_allclose(result, result.T, atol=1e-15)
    assert np.linalg.eigvalsh(result).min() > 0


def test_bfgs_does_not_solve_linear_system(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("BFGS must not solve a linear system")
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    r = bfgs((7, 5), fun=quadratic, grad=quad_gradient, omega=.1)
    assert r.converged


@pytest.mark.parametrize("y", [[0, 0], [-1, 0], [1e-20, 1]])
def test_unsafe_curvature_is_rejected(y):
    with pytest.raises(FloatingPointError):
        inverse_bfgs_update(np.eye(2), [1, 0], y)


def test_negative_curvature_failure_and_explicit_skip():
    fun = lambda x: -.5*x@x
    grad = lambda x: -x
    r = bfgs((1, 0), fun=fun, grad=grad, step_mode="fixed")
    assert r.status == "numerical_failure" and r.nit == 1
    assert r.history[-1]["update_action"] == "failed_curvature"
    r = bfgs((1, 0), fun=fun, grad=grad, step_mode="fixed", curvature_action="skip", maxit=2)
    assert r.status == "max_iterations"
    assert all(h["update_action"] == "skipped_curvature" for h in r.history[1:])


def test_singular_hessian_safeguard_is_explicit():
    kw = dict(fun=lambda x: .5*x@x, grad=lambda x: x, hess=lambda x: np.zeros((2, 2)))
    r = newton((1, 2), **kw)
    assert r.status == "numerical_failure"
    r = newton((1, 2), fallback="steepest_descent", **kw)
    assert r.converged and r.history[1]["step_kind"] == "fallback_condition"


def test_non_descent_newton_is_not_accepted_by_armijo():
    kw = dict(fun=lambda x: -.5*x@x, grad=lambda x: -x, hess=lambda x: -np.eye(2))
    r = newton((1, 2), **kw)
    assert r.status == "line_search_failure" and r.nit == 0
    r = newton((1, 2), step_mode="fixed", **kw)
    assert r.converged and r.nit == 1  # stationary maximum, never called a minimum


def test_armijo_backtracks_and_never_accepts_non_descent_or_exhaustion():
    f = lambda x: .5*x@x
    x = np.array([1., 0.])
    a = armijo(f, x, f(x), x, -10*x)
    assert a == .125
    assert armijo(f, x, f(x), x, x) is None
    assert armijo(f, x, f(x), x, -10*x, max_backtracks=0) is None


@pytest.mark.parametrize("solver", [newton, bfgs])
def test_iteration_budget_and_initial_convergence(solver):
    r = solver(TARGET, fun=quadratic, grad=quad_gradient, maxit=0)
    assert r.converged and r.nit == 0
    r = solver((7, 5), fun=quadratic, grad=quad_gradient, maxit=0)
    assert r.status == "max_iterations" and len(r.history) == 1


@pytest.mark.parametrize("x", [[np.nan, 0], [np.inf, 0], [1e3, 1e3]])
@pytest.mark.parametrize("solver", [newton, bfgs])
def test_nonfinite_or_overflowing_initial_point_returns_failure(solver, x):
    r = solver(x)
    assert r.status == "numerical_failure" and not r.converged


@pytest.mark.parametrize("kwargs", [{"tol":0}, {"maxit":-1}, {"maxit":1.5}, {"alpha":0}])
@pytest.mark.parametrize("solver", [newton, bfgs])
def test_invalid_controls_rejected(solver, kwargs):
    with pytest.raises(ValueError):
        solver((0, 0), **kwargs)


def test_finite_differences_independent_vector_example():
    f = lambda x: np.array([x[0]**2+x[1], x[0]*x[1]])
    assert_allclose(finite_difference_jacobian(f, [2., 3.]), [[4, 1], [3, 2]], atol=1e-9)


def test_armijo_rejects_overflowed_trials_then_recovers():
    fun = lambda x: float(np.exp(x).sum())
    x = np.array([10., 0.])
    g = np.exp(x)
    p = np.array([-1., 1000.])  # descent overall, but alpha=1 overflows exp(x2)
    a = armijo(fun, x, fun(x), g, p)
    assert a is not None and 0 < a < 1
    assert fun(x+a*p) <= fun(x) + 1e-4*a*(g@p)


@pytest.mark.parametrize("solver", [newton, bfgs])
def test_solver_reports_exhausted_line_search(solver):
    # Deliberately inconsistent function/gradient: no trial can satisfy Armijo.
    r = solver((1, 0), fun=lambda x: 1., grad=lambda x: np.ones(2), max_backtracks=2)
    assert r.status == "line_search_failure" and r.nit == 0


def test_newton_accepts_numerically_approximated_hessian():
    from sc_cw1.exercise1.model import numerical_hessian, gradient
    r = newton((-2, 1), hess=numerical_hessian)
    assert r.converged and np.linalg.norm(gradient(r.x)) < 1e-6


def test_ill_conditioned_hessian_fails_at_explicit_condition_limit():
    r = newton((1, 0), fun=lambda x: .5*x@x, grad=lambda x: x,
               hess=lambda x: np.diag([1., 1e-13]))
    assert r.status == "numerical_failure" and "conditioned" in r.message


@pytest.mark.parametrize("solver", [newton, bfgs])
def test_nonfinite_fixed_trial_preserves_last_valid_iterate(solver):
    kw = dict(fun=lambda x: .5*x@x, grad=lambda x: x, step_mode="fixed")
    if solver is newton:
        kw["hess"] = lambda x: 1e-200*np.eye(2)
    else:
        kw["omega"] = 1e200
    r = solver((1, 2), **kw)
    assert r.status == "numerical_failure" and r.nit == 0
    assert_allclose(r.x, [1, 2])


def test_initial_input_is_not_mutated_and_history_owns_its_data():
    x0 = np.array([-2., 1.])
    before = x0.copy()
    r = newton(x0)
    assert_allclose(x0, before)
    x0[:] = 123
    assert_allclose(r.iterates[0], before)


def test_strict_tolerance_equality_does_not_converge():
    for solver in (newton, bfgs):
        r = solver((1, 0), fun=lambda x: .5*x@x, grad=lambda x: x, tol=1., maxit=0)
        assert not r.converged and r.status == "max_iterations"
