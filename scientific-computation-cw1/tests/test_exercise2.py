"""Residual derivatives, exact update examples and prescribed experiment controls."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sc_cw1.common.utilities import finite_difference_jacobian
from sc_cw1.exercise2.model import F, jacobian, phi, grad_phi, hessian_phi, residual_grid, phi_grid
from sc_cw1.exercise2.part_d_gradient_descent import gradient_descent
from sc_cw1.exercise2.part_e_momentum import momentum_descent
from sc_cw1.exercise2.part_f_comparison import compare
from sc_cw1.exercise2.part_b_contours import estimate_minima
from sc_cw1.exercise2.part_c_intersections import graphical_estimates, refine_intersections, polyline_intersections


@pytest.mark.parametrize("x", [(0, -1), (.3, .7), (1.4, -.9), (-2, 2), (2, -2)])
def test_residual_and_jacobian_derivatives(x):
    assert_allclose(jacobian(x), finite_difference_jacobian(F, x), atol=2e-9, rtol=2e-9)
    assert_allclose(grad_phi(x), finite_difference_jacobian(phi, x)[0], atol=2e-8, rtol=2e-8)
    assert_allclose(hessian_phi(x), finite_difference_jacobian(grad_phi, x), atol=3e-8, rtol=3e-8)
    assert phi(x) == pytest.approx(np.linalg.norm(F(x))**2)
    assert_allclose(grad_phi(x), 2*jacobian(x).T@F(x))


def test_hand_computable_coursework_initial_values_and_first_step():
    assert_allclose(F((0, -1)), [1, 2])
    assert_allclose(jacobian((0, -1)), [[-2, -2], [0, -1]])
    assert phi((0, -1)) == 5
    assert_allclose(grad_phi((0, -1)), [-4, -8])
    r = gradient_descent(maxit=1)
    assert_allclose(r.iterates, [[0, -1], [.2, -.6]])
    assert r.status == "max_iterations"
    assert r.parameters["step_size"] == .05


def test_mesh_evaluations_match_scalar_functions():
    x = np.array([-.5, 1, 2])
    y = np.array([.1, -.3, 1])
    assert_allclose(np.column_stack(residual_grid(x, y)), [F(p) for p in zip(x, y)])
    assert_allclose(phi_grid(x, y), [phi(p) for p in zip(x, y)])


def test_two_hand_computable_gradient_steps_on_identity_residual():
    r = gradient_descent((1, -2), step_size=.1, maxit=2,
                          residual=lambda x: x, jac=lambda x: np.eye(2))
    assert_allclose(r.iterates, [[1, -2], [.8, -1.6], [.64, -1.28]])


def test_two_hand_computable_momentum_steps_and_zero_initial_memory():
    r = momentum_descent((1, -2), gamma=.5, step_size=.1, maxit=2,
                          residual=lambda x: x, jac=lambda x: np.eye(2))
    assert_allclose(r.iterates, [[1, -2], [.8, -1.6], [.54, -1.08]])
    assert_allclose([r.history[2]["direction1"], r.history[2]["direction2"]], [2.6, -5.2])
    assert r.parameters["initial_direction1"] == r.parameters["initial_direction2"] == 0


def test_custom_initial_momentum_is_used_without_reset():
    r = momentum_descent((1, -2), gamma=.5, initial_direction=(2, 4), step_size=.1,
                          maxit=1, residual=lambda x: x, jac=lambda x: np.eye(2))
    assert_allclose(r.iterates[1], [.7, -1.8])


@pytest.mark.parametrize("solver", [gradient_descent, momentum_descent])
def test_residual_stopping_including_final_allowed_update(solver):
    kw = dict(residual=lambda x: x, jac=lambda x: np.eye(2))
    r = solver((0, 0), maxit=0, **kw)
    assert r.converged and r.nit == 0
    r = solver((1, -2), step_size=.5, maxit=1, **kw)
    assert r.converged and r.nit == 1
    r = solver((1, -2), maxit=0, **kw)
    assert r.status == "max_iterations" and len(r.history) == 1


def test_nonroot_stationary_residual_minimum_is_not_convergence():
    # ∇φ=0 but F is nonzero: gradient stopping would wrongly report success.
    r = gradient_descent(residual=lambda x: np.ones(2), jac=lambda x: np.zeros((2, 2)), maxit=3)
    assert r.status == "max_iterations" and r.nit == 3
    assert r.history[-1]["residual_norm"] == pytest.approx(np.sqrt(2))


def test_coursework_runs_use_exact_budget_settings_and_fresh_initial_state():
    runs = compare()
    assert set(runs) == {.1, .3, .7}
    for gamma, r in runs.items():
        assert_allclose(r.iterates[0], [0, -1])
        assert_allclose(r.iterates[1], [.2, -.6])
        assert r.parameters["gamma"] == gamma
        assert r.parameters["step_size"] == .05
        assert r.parameters["tol"] == 1e-6 and r.parameters["maxit"] == 200
        assert r.nit <= 200 and len(r.history) == r.nit+1
        assert all(row["alpha"] == .05 for row in r.history[1:])
        if r.converged:
            assert np.linalg.norm(F(r.x)) < 1e-6
        else:
            assert r.status == "max_iterations" and r.nit == 200
            assert np.linalg.norm(F(r.x)) >= 1e-6


@pytest.mark.parametrize("solver", [gradient_descent, momentum_descent])
def test_numerical_failure_is_explicit(solver):
    r = solver((np.nan, 0))
    assert r.status == "numerical_failure"
    r = solver((1e100, 0))
    assert r.status == "numerical_failure"


@pytest.mark.parametrize("gamma", [0, 1, -1, np.nan])
def test_invalid_momentum_coefficient(gamma):
    with pytest.raises(ValueError):
        momentum_descent(gamma=gamma)


def test_polyline_graphical_estimates_and_separate_refinement():
    _, _, estimates = graphical_estimates()
    assert len(estimates) >= 2
    for x in estimates:
        assert np.linalg.norm(F(x)) < 1e-5  # finite plotted-curve resolution
    refined = refine_intersections(estimates)
    assert len(refined) == len(estimates)
    assert all(r["status"] == "converged" and r["residual_norm"] < 1e-12 for r in refined)
    crossing = polyline_intersections(np.array([[0, 0], [1, 1]]), np.array([[0, 1], [1, 0]]))
    assert_allclose(crossing, [[.5, .5]])


def test_apparent_minima_check_curvature_and_residual_separately():
    candidates, runs, points = estimate_minima(201)
    assert len(candidates) == len(runs) and len(points) >= 2
    for p in points:
        x = (p["x1"], p["x2"])
        assert p["classification"] == "minimum"
        assert np.linalg.norm(grad_phi(x)) < 1e-7
        assert p["is_root"] == (np.linalg.norm(F(x)) < 1e-6)


def test_failed_descent_trial_preserves_finite_history():
    r = gradient_descent((1, -2), step_size=1e308, residual=lambda x: x,
                          jac=lambda x: np.eye(2))
    assert r.status == "numerical_failure" and r.nit == 0
    assert_allclose(r.x, [1, -2])


def test_momentum_runs_do_not_share_memory():
    first = momentum_descent(gamma=.3)
    momentum_descent(gamma=.7)
    repeated = momentum_descent(gamma=.3)
    assert_allclose(first.iterates, repeated.iterates, atol=0, rtol=0)


def test_descent_strict_tolerance_boundary():
    r = gradient_descent((1, 0), tol=1., maxit=0, residual=lambda x: x, jac=lambda x: np.eye(2))
    assert not r.converged and r.status == "max_iterations"
