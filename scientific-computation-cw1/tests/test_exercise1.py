"""Independent objective/derivative checks and coursework-specific requirements."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sc_cw1.common.utilities import finite_difference_jacobian
from sc_cw1.exercise1 import model
from sc_cw1.exercise1.part_b_newton import newton
from sc_cw1.exercise1.part_c_minima import classify_stationary_point, distinct_stationary_points, multistart
from sc_cw1.exercise1.part_e_comparison import compare

POINTS = [(0., 0.), (.3, -.7), (-1.2, .8), (1.6, -1.4), (-2., 1.)]


def test_objective_known_values():
    assert model.objective((0, 0)) == .5
    assert_allclose(model.objective((1, 0)), np.exp(.5)+.5)
    assert_allclose(model.objective((.5, -.5)), np.exp(.25)+np.sin(.5)**2)
    assert_allclose(model.gradient((0, 0)), [1, -1], atol=0)
    assert_allclose(model.hessian((0, 0)), [[1, 2], [2, 1]], atol=0)


@pytest.mark.parametrize("x", POINTS)
def test_analytical_gradient_against_objective_central_difference(x):
    independent = finite_difference_jacobian(model.objective, x)[0]
    assert_allclose(model.gradient(x), independent, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("x", POINTS)
def test_hessian_against_two_independent_differences(x):
    x = np.array(x)
    H = model.hessian(x)
    assert_allclose(H, H.T, atol=1e-14)
    assert_allclose(H, model.numerical_hessian(x), atol=2e-7, rtol=2e-7)
    # Direct second differences of the *objective*, independent of ∇J as well.
    h = 2e-4
    e1, e2 = np.eye(2)*h
    f = model.objective
    direct = np.array([[(f(x+e1)-2*f(x)+f(x-e1))/h**2,
                        (f(x+e1+e2)-f(x+e1-e2)-f(x-e1+e2)+f(x-e1-e2))/(4*h*h)],
                       [0., (f(x+e2)-2*f(x)+f(x-e2))/h**2]])
    direct[1, 0] = direct[0, 1]
    assert_allclose(H, direct, rtol=4e-6, atol=4e-6)


def test_vectorised_objective_uses_identical_equation():
    points = np.array(POINTS)
    assert_allclose(model.objective_grid(points[:, 0], points[:, 1]),
                    [model.objective(x) for x in points])


@pytest.mark.parametrize("x0", [(-2, 1), (0, 0), (2, -2), (1, 1)])
def test_converged_newton_points_and_armijo_inequality(x0):
    r = newton(x0, fallback="steepest_descent")
    assert r.converged, r.message
    assert np.linalg.norm(model.gradient(r.x)) < r.parameters["tol"]
    assert len(r.history) == r.nit+1
    for before, after in zip(r.history, r.history[1:]):
        x = np.array([before["x1"], before["x2"]])
        p = np.array([after["direction1"], after["direction2"]])
        assert model.gradient(x) @ p < 0
        bound = before["value"]+r.parameters["c1"]*after["alpha"]*(model.gradient(x)@p)
        assert after["value"] <= bound


def test_fixed_coursework_comparison_has_no_safeguards():
    runs = compare()
    for r in runs.values():
        assert_allclose(r.iterates[0], [-2, 1])
        assert r.parameters["step_mode"] == "fixed"
        assert all(row["alpha"] == 1 for row in r.history[1:])
        assert all(not row["step_kind"].startswith("fallback") for row in r.history)
    assert runs["BFGS"].parameters["omega"] == .01
    assert runs["BFGS"].parameters["curvature_action"] == "fail"
    r = runs["Newton"]
    expected = np.array([-2., 1.]) + np.linalg.solve(model.hessian((-2, 1)), -model.gradient((-2, 1)))
    assert_allclose(r.iterates[1], expected)


@pytest.mark.parametrize("diagonal, label", [([1, 2], "minimum"), ([-1, -2], "maximum"),
                                            ([-1, 2], "saddle"), ([0, 2], "inconclusive")])
def test_stationary_classification(diagonal, label):
    info = classify_stationary_point([0, 0], grad=lambda x: np.zeros(2),
                                     hess=lambda x: np.diag(diagonal))
    assert info["classification"] == label


def test_nonstationary_point_is_never_a_minimum():
    assert classify_stationary_point((0, 0))["classification"] == "not_stationary"


def test_deduplication_excludes_failed_and_nonstationary_runs():
    from sc_cw1.common.utilities import SolverResult
    def fake(x, status):
        r = SolverResult(status=status)
        r.append(x, 0., np.linalg.norm(x))
        return r
    runs = [fake((0, 0), "converged"), fake((1e-9, 0), "converged"),
            fake((1, 1), "max_iterations"), fake((2, 2), "converged")]
    pts = distinct_stationary_points(runs, fun=lambda x: float(x@x),
        grad=lambda x: 2*x, hess=lambda x: 2*np.eye(2))
    assert len(pts) == 1
    assert pts[0]["runs_in_cluster"] == 2


def test_multistart_spans_square_and_classifies_only_convergence():
    starts, runs, points = multistart(5)
    assert len(starts) == len(runs) == 25
    assert (-2., -2.) in starts and (2., 2.) in starts
    assert points
    for p in points:
        assert p["status"] == "converged"
        if p["classification"] == "minimum":
            assert p["gradient_norm"] < 1e-6
            assert p["eigenvalue_min"] > 0
    assert sum(p["lowest_minimum_found"] for p in points) == 1
