"""Simple independent examples and essential numerical failure checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sc_cw1.exercise1.part_b_newton import newton, armijo
from sc_cw1.exercise1.part_d_bfgs import bfgs, inverse_bfgs_update

A = np.array([[4., 1.], [1., 3.]])
TARGET = np.array([1., -2.])


def quadratic(x):
    z = x-TARGET
    return .5*z @ A @ z


def quad_gradient(x):
    return A @ (x-TARGET)


def test_newton_quadratic_in_one_step():
    r = newton((7, 5), fun=quadratic, grad=quad_gradient, hess=lambda x: A, tol=1e-12)
    assert r["status"] == "converged" and r["iterations"] == 1
    assert_allclose(r["x"], TARGET, atol=1e-14)


@pytest.mark.parametrize("line_search", [False, True])
def test_bfgs_quadratic_without_linear_solve(line_search, monkeypatch):
    def forbidden(*args):
        raise AssertionError("BFGS must not solve a linear system")
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    r = bfgs((7, 5), fun=quadratic, grad=quad_gradient, omega=.1,
             tol=1e-9, line_search=line_search)
    assert r["status"] == "converged"
    assert_allclose(r["x"], TARGET, atol=1e-9)


def test_inverse_bfgs_secant_symmetry_and_positive_curvature():
    B = np.array([[.7, .1], [.1, .5]])
    s = np.array([.3, -.2])
    y = A @ s
    updated = inverse_bfgs_update(B, s, y)
    assert_allclose(updated @ y, s, atol=1e-15)
    assert_allclose(updated, updated.T, atol=1e-15)
    assert updated[0, 0] > 0 and np.linalg.det(updated) > 0
    for unsafe in [np.zeros(2), -s]:
        with pytest.raises(ValueError):
            inverse_bfgs_update(B, s, unsafe)


def test_armijo_backtracking_and_failure():
    f = lambda x: .5*x @ x
    x = np.array([1., 0.])
    assert armijo(f, x, f(x), x, -10*x) == .125
    assert armijo(f, x, f(x), x, x) is None
    assert armijo(f, x, f(x), x, -10*x, max_backtracks=0) is None


def test_singular_hessian_and_non_descent_are_not_silently_accepted():
    kw = dict(fun=lambda x: .5*x @ x, grad=lambda x: x, hess=lambda x: np.zeros((2, 2)))
    r = newton((1, 2), **kw)
    assert r["status"] == "numerical_failure" and r["iterations"] == 0
    assert_allclose(r["x"], [1, 2])
    kw["hess"] = lambda x: -np.eye(2)
    r = newton((1, 2), **kw)
    assert r["status"] == "line_search_failure" and r["iterations"] == 0
    assert_allclose(r["x"], [1, 2])


@pytest.mark.parametrize("solver", [newton, bfgs])
def test_initial_convergence_budget_and_nonfinite_values(solver):
    r = solver(TARGET, fun=quadratic, grad=quad_gradient, maxit=0)
    assert r["status"] == "converged" and r["iterations"] == 0
    r = solver((7, 5), fun=quadratic, grad=quad_gradient, maxit=0)
    assert r["status"] == "max_iterations" and r["iterations"] == 0
    assert solver((np.nan, 0))["status"] == "numerical_failure"
    r = solver((0, 0), fun=lambda x: np.nan, grad=lambda x: np.zeros(2))
    assert r["status"] == "numerical_failure"


@pytest.mark.parametrize("solver", [newton, bfgs])
@pytest.mark.parametrize("maxit", [0, 1, 2])
def test_counter_stops_at_the_allowed_number_of_accepted_updates(solver, maxit):
    r = solver((-2, 1), maxit=maxit)
    assert r["status"] == "max_iterations" and r["iterations"] == maxit
    assert len(r["iterates"]) == maxit+1
    assert len(r["directions"]) == len(r["alphas"]) == maxit+1


@pytest.mark.parametrize("solver", [newton, bfgs])
def test_tolerance_equality_and_convergence_on_last_allowed_update(solver):
    kw = dict(fun=lambda x: .5*x @ x, grad=lambda x: x)
    if solver is newton:
        kw["hess"] = lambda x: np.eye(2)
    # Equality does not pass the strict tolerance test; one update is needed.
    r = solver((1, 0), tol=1., maxit=1, **kw)
    assert r["status"] == "converged" and r["iterations"] == 1
    assert_allclose(r["iterates"], [[1, 0], [0, 0]])


def test_unsafe_bfgs_curvature_is_a_reported_failure():
    r = bfgs((1, 0), fun=lambda x: -.5*x @ x, grad=lambda x: -x, line_search=False)
    assert r["status"] == "numerical_failure" and r["iterations"] == 1
