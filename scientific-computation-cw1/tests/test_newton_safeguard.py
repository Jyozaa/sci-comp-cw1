"""Focused checks for the opt-in direction safeguard and unchanged strict mode."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sc_cw1.exercise1 import part_b_newton as solver
from sc_cw1.exercise1.model import objective, gradient, hessian


def test_origin_fallback_is_labelled_and_every_accepted_step_passes_armijo():
    strict = solver.newton((0, 0))
    assert strict["status"] == "line_search_failure" and strict["iterations"] == 0
    result = solver.newton((0, 0), fallback=True)
    assert result["status"] == "converged"
    assert result["step_kinds"][1] == "gradient_fallback"
    assert_array_equal(result["directions"][1], -gradient((0, 0)))
    for k in range(1, len(result["iterates"])):
        x, delta, alpha = result["iterates"][k-1], result["directions"][k], result["alphas"][k]
        g = gradient(x)
        assert g @ delta < 0
        if result["step_kinds"][k] == "gradient_fallback":
            assert_array_equal(delta, -g)
        else:
            assert_allclose(hessian(x) @ delta, -g, rtol=1e-12, atol=1e-12)
        assert result["values"][k] <= result["values"][k-1]+1e-4*alpha*(g @ delta)


@pytest.mark.parametrize("H", [np.zeros((2, 2)), np.diag([1., 1e-14]),
                              np.full((2, 2), np.nan), -np.eye(2)])
def test_unusable_hessian_or_direction_can_use_negative_gradient(H):
    result = solver.newton((1, 2), fun=lambda x: .5*x @ x, grad=lambda x: x,
                           hess=lambda x: H, fallback=True)
    assert result["status"] == "converged" and result["iterations"] == 1
    assert result["step_kinds"] == ["initial", "gradient_fallback"]
    assert_array_equal(result["directions"][1], [-1, -2])
    assert_array_equal(result["x"], [0, 0])


@pytest.mark.parametrize("bad_direction", [np.full(2, np.nan), np.full(2, 1e308)])
def test_invalid_solve_direction_is_rejected_before_backtracking(monkeypatch, bad_direction):
    monkeypatch.setattr(np.linalg, "solve", lambda H, g: bad_direction.copy())
    result = solver.newton((2, -2), fun=lambda x: .5*x @ x, grad=lambda x: x,
                           hess=lambda x: np.eye(2), fallback=True)
    assert result["status"] == "converged"
    assert result["step_kinds"][1] == "gradient_fallback"
    assert_array_equal(result["directions"][1], [-2, 2])


def test_exhausted_newton_backtracking_can_retry_once_with_negative_gradient():
    # This approximate Hessian gives an enormous descent step: 50 halvings fail.
    kw = dict(fun=lambda x: .5*x @ x, grad=lambda x: x,
              hess=lambda x: 1e-20*np.eye(2))
    strict = solver.newton((1, 2), **kw)
    assert strict["status"] == "line_search_failure" and strict["iterations"] == 0
    result = solver.newton((1, 2), fallback=True, **kw)
    assert result["status"] == "converged" and result["iterations"] == 1
    assert result["step_kinds"][1] == "gradient_fallback"
    assert result["alphas"][1] == 1.
    assert_array_equal(result["x"], [0, 0])


@pytest.mark.parametrize("H,attempts", [(-np.eye(2), 1), (1e-20*np.eye(2), 2)])
def test_failed_backtracking_never_accepts_a_step_or_retries_gradient_twice(monkeypatch, H, attempts):
    calls = []
    def reject(*args):
        calls.append(args[-1].copy())
        return None
    monkeypatch.setattr(solver, "armijo", reject)
    result = solver.newton((1, 2), fun=lambda x: .5*x @ x, grad=lambda x: x,
                           hess=lambda x: H, fallback=True)
    assert result["status"] == "line_search_failure" and result["iterations"] == 0
    assert len(calls) == attempts
    assert_array_equal(calls[-1], [-1, -2])
    assert len(result["iterates"]) == 1 and result["step_kinds"] == ["initial"]


def test_fixed_step_ignores_fallback_including_singular_and_non_descent_cases(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Fixed-step Newton must not call Armijo")
    monkeypatch.setattr(solver, "armijo", forbidden)
    variants = [{}, dict(fun=lambda x: .5*x @ x, grad=lambda x: x,
                         hess=lambda x: np.zeros((2, 2)), maxit=1),
                dict(fun=lambda x: .5*x @ x, grad=lambda x: x,
                     hess=lambda x: -np.eye(2), maxit=1)]
    for kw in variants:
        strict = solver.newton((-2, 1), line_search=False, **kw)
        safeguarded = solver.newton((-2, 1), line_search=False, fallback=True, **kw)
        for key in strict:
            assert_array_equal(strict[key], safeguarded[key])
        assert "gradient_fallback" not in safeguarded["step_kinds"]


def test_fallback_preserves_initial_stops_budget_and_nonfinite_failure():
    kw = dict(fun=lambda x: .5*x @ x, grad=lambda x: x, hess=lambda x: np.eye(2))
    assert solver.newton((0, 0), fallback=True, maxit=0, **kw)["status"] == "converged"
    result = solver.newton((1, 2), fallback=True, maxit=0, **kw)
    assert result["status"] == "max_iterations" and result["iterations"] == 0
    assert solver.newton((np.nan, 0), fallback=True)["status"] == "numerical_failure"
    result = solver.newton((1, 2), fallback=True, maxit=1, **kw)
    assert result["status"] == "converged" and result["iterations"] == 1
