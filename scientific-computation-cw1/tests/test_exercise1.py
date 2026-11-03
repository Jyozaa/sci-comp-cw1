"""Objective/derivative checks and the required Exercise 1 settings."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sc_cw1.exercise1 import model, part_b_newton, part_d_bfgs
from sc_cw1.exercise1.part_c_minima import multistart, distinct_stationary_points
from sc_cw1.exercise1.part_e_comparison import compare


def test_known_objective_values():
    assert model.objective((0, 0)) == .5
    assert_allclose(model.objective((1, 0)), np.exp(.5)+.5)
    assert_allclose(model.gradient((0, 0)), [1, -1])
    assert_allclose(model.hessian((0, 0)), [[1, 2], [2, 1]])


@pytest.mark.parametrize("point", [(0., 0.), (.3, -.7), (-1.2, .8), (1.6, -1.4), (-2., 1.)])
def test_analytical_derivatives_against_finite_differences(point):
    x = np.array(point)
    h = 1e-5
    numerical_gradient = np.zeros(2)
    for j in range(2):
        e = np.eye(2)[j]*h
        numerical_gradient[j] = (model.objective(x+e)-model.objective(x-e))/(2*h)
    assert_allclose(model.gradient(x), numerical_gradient, rtol=2e-8, atol=2e-8)
    H = model.hessian(x)
    assert_allclose(H, H.T, atol=1e-14)
    assert_allclose(H, model.numerical_hessian(x), rtol=2e-7, atol=2e-7)
    # Independent second differences of J also check the gradient-based Hessian.
    h = 2e-4
    e1, e2 = np.eye(2)*h
    f = model.objective
    diagonal = [(f(x+e)-2*f(x)+f(x-e))/h**2 for e in (e1, e2)]
    mixed = (f(x+e1+e2)-f(x+e1-e2)-f(x-e1+e2)+f(x-e1-e2))/(4*h*h)
    assert_allclose(H, [[diagonal[0], mixed], [mixed, diagonal[1]]], rtol=4e-6, atol=4e-6)
    assert_allclose(model.objective_grid(x[0], x[1]), f(x))


@pytest.mark.parametrize("x0", [(-2, 1), (0, 0), (2, -2), (1, 1)])
def test_newton_convergence_and_every_armijo_step(x0):
    r = part_b_newton.newton(x0, fallback=True)
    assert r["status"] == "converged"
    assert np.linalg.norm(model.gradient(r["x"])) < 1e-6
    for k in range(1, len(r["iterates"])):
        x, p, a = r["iterates"][k-1], r["directions"][k], r["alphas"][k]
        assert model.gradient(x) @ p < 0
        assert r["values"][k] <= r["values"][k-1]+1e-4*a*(model.gradient(x) @ p)
    if x0 == (0, 0):
        assert "fallback_non_descent" in r["step_kinds"]


def test_prescribed_comparison_has_fixed_unit_steps_and_no_armijo(monkeypatch):
    def forbidden(*args):
        raise AssertionError("1(e) must not call line search")
    monkeypatch.setattr(part_b_newton, "armijo", forbidden)
    monkeypatch.setattr(part_d_bfgs, "armijo", forbidden)
    runs = compare()
    for r in runs.values():
        assert r["status"] == "converged"
        assert_allclose(r["iterates"][0], [-2, 1])
        assert np.all(r["alphas"][1:] == 1)
    assert set(runs["Newton"]["step_kinds"][1:]) == {"newton"}
    assert_allclose(runs["BFGS"]["directions"][1], -.01*model.gradient((-2, 1)))


def test_multistart_counts_only_converged_distinct_points():
    starts, runs, points = multistart()
    assert len(starts) == len(runs) == 441
    assert (-2., -2.) in starts and (2., 2.) in starts
    assert sum(p["classification"] == "minimum" for p in points) == 2
    assert sum(p["classification"] == "saddle" for p in points) == 1
    assert sum(p["lowest_minimum_found"] for p in points) == 1
    good = next(r for r in runs if r["status"] == "converged")
    bad = dict(good, status="max_iterations")
    clustered = distinct_stationary_points([good, good, bad])
    assert len(clustered) == 1 and clustered[0]["runs_in_cluster"] == 2
