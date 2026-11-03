"""Residual derivatives, hand-computable updates and required Exercise 2 settings."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sc_cw1.exercise2.model import F, jacobian, phi, grad_phi, phi_grid
from sc_cw1.exercise2.part_d_gradient_descent import gradient_descent
from sc_cw1.exercise2.part_e_momentum import momentum_descent
from sc_cw1.exercise2.part_f_comparison import compare
from sc_cw1.exercise2.part_b_contours import grid_minimum_candidates
from sc_cw1.exercise2.part_c_intersections import graphical_estimates


@pytest.mark.parametrize("point", [(0, -1), (.3, .7), (1.4, -.9), (-2, 2), (2, -2)])
def test_jacobian_and_residual_gradient(point):
    x = np.array(point, dtype=float)
    numerical_J, numerical_g = np.zeros((2, 2)), np.zeros(2)
    h = 1e-5
    for j in range(2):
        e = np.eye(2)[j]*h
        numerical_J[:, j] = (F(x+e)-F(x-e))/(2*h)
        numerical_g[j] = (phi(x+e)-phi(x-e))/(2*h)
    assert_allclose(jacobian(x), numerical_J, atol=2e-9, rtol=2e-9)
    assert_allclose(grad_phi(x), numerical_g, atol=2e-8, rtol=2e-8)
    assert_allclose(grad_phi(x), 2*jacobian(x).T @ F(x))
    assert phi(x) == pytest.approx(np.linalg.norm(F(x))**2)
    assert_allclose(phi_grid(x[0], x[1]), phi(x))


def test_required_initial_values_and_first_step():
    assert_allclose(F((0, -1)), [1, 2])
    assert_allclose(jacobian((0, -1)), [[-2, -2], [0, -1]])
    assert_allclose(grad_phi((0, -1)), [-4, -8])
    assert phi((0, -1)) == 5
    r = gradient_descent(maxit=1)
    assert_allclose(r["iterates"], [[0, -1], [.2, -.6]])
    assert r["status"] == "max_iterations" and r["alphas"][1] == .05


def test_two_hand_computable_ordinary_and_momentum_steps():
    kw = dict(residual=lambda x: x, jac=lambda x: np.eye(2), step_size=.1, maxit=2)
    ordinary = gradient_descent((1, -2), **kw)
    momentum = momentum_descent((1, -2), gamma=.5, **kw)
    assert_allclose(ordinary["iterates"], [[1, -2], [.8, -1.6], [.64, -1.28]])
    assert_allclose(momentum["iterates"], [[1, -2], [.8, -1.6], [.54, -1.08]])
    assert_allclose(momentum["directions"][0], [0, 0])
    assert_allclose(momentum["directions"][2], [2.6, -5.2])


@pytest.mark.parametrize("solver", [gradient_descent, momentum_descent])
def test_residual_stopping_and_exact_update_budget(solver):
    # Nonzero F with zero gradient must exhaust 200 updates, not claim convergence.
    r = solver(residual=lambda x: np.ones(2), jac=lambda x: np.zeros((2, 2)))
    assert r["status"] == "max_iterations" and r["iterations"] == 200
    assert len(r["iterates"]) == 201
    kw = dict(residual=lambda x: x, jac=lambda x: np.eye(2))
    assert solver((0, 0), maxit=0, **kw)["status"] == "converged"
    assert solver((1, -2), step_size=.5, maxit=1, **kw)["status"] == "converged"
    assert solver((np.nan, 0))["status"] == "numerical_failure"


def test_prescribed_momentum_values_and_fresh_initial_state():
    runs = compare()
    assert set(runs) == {.1, .3, .7}
    for r in runs.values():
        assert_allclose(r["iterates"][0], [0, -1])
        assert_allclose(r["iterates"][1], [.2, -.6])
        assert np.all(r["alphas"][1:] == .05)
        assert r["status"] == "converged" and r["iterations"] <= 200
        assert np.linalg.norm(F(r["x"])) < 1e-6
        assert len(r["iterates"]) == r["iterations"]+1


def test_graphical_and_contour_estimates_have_appropriate_precision():
    _, _, estimates = graphical_estimates()
    assert len(estimates) == 2
    for p in estimates:
        assert np.linalg.norm(F(p)) < 1e-5
    candidates = grid_minimum_candidates(201)
    assert len(candidates) == 2
    h = .02
    for p in candidates:
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            assert phi(p) < phi(p+h*np.array([dx,dy]))
