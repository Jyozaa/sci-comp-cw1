"""Review 1 additions: reduction, independent derivatives and rigorous interval proof.

The pre-review tests are unchanged. Float comparisons test implementation
regressions; the proof's root exclusions/orderings use exact rationals only.
"""

import copy
import csv
from fractions import Fraction
from io import StringIO

import numpy as np
from numpy.testing import assert_allclose
import pytest

from sc_cw1.exercise1 import model
from sc_cw1.exercise1.global_analysis import (
    Interval, ProofFailure, derivative_bounds, exp_point, from_uv,
    global_minimum_proof, objective_bounds, outward, reduced_derivative,
    reduced_objective, reduced_second_derivative, sin_cos_interval, sin_cos_point,
    to_uv, transformed_objective, verify_cover,
)


@pytest.mark.parametrize("x", [(0, 0), (-2, 1), (.3, -.7), (1.5, 1.5), (-1.2, 1.2)])
def test_orthogonal_transform_and_original_objective(x):
    uv = to_uv(x)
    assert_allclose(from_uv(uv), x, atol=1e-15)
    assert_allclose(uv@uv, np.array(x)@np.array(x), atol=3e-15)
    assert_allclose(transformed_objective(uv), model.objective(x), atol=4e-14, rtol=2e-14)


@pytest.mark.parametrize("a", [-2., -1.3, -.742, -.659, 0., .4, 1.5])
def test_reduced_objective_derivatives_and_chain_rule(a):
    assert_allclose(reduced_objective(a), model.objective((a, -a)), atol=1e-14)
    direction = np.array([1., -1.])
    assert_allclose(reduced_derivative(a), model.gradient((a, -a))@direction, atol=2e-13)
    assert_allclose(reduced_second_derivative(a), direction@model.hessian((a, -a))@direction,
                    atol=4e-13)
    h = 1e-5
    first = (reduced_objective(a+h)-reduced_objective(a-h))/(2*h)
    second = (reduced_derivative(a+h)-reduced_derivative(a-h))/(2*h)
    assert_allclose(reduced_derivative(a), first, rtol=3e-8, atol=3e-8)
    assert_allclose(reduced_second_derivative(a), second, rtol=3e-8, atol=3e-8)


@pytest.mark.parametrize("u", [-2., -.3, 0., 1.5])
def test_strict_increase_with_v_squared(u):
    t = np.linspace(0, 3, 51)
    values = [transformed_objective((u, np.sqrt(v2))) for v2 in t]
    assert np.all(np.diff(values) > 0)
    derivative = .5*np.exp((u*u+t)/2)+1+np.sin(2*(t-u*u))
    assert np.all(derivative >= .5*np.exp((u*u+t)/2))


def test_sign_preference_and_analytic_tail_bound():
    for a in [.1, .6, 1.2, 2., 3.]:
        assert_allclose(reduced_objective(a)-reduced_objective(-a), 4*a, atol=1e-11)
    # Exact inequalities used in the proof: exp(4)>1+4+4²/2=13.
    assert exp_point(Fraction(4)).lo > 13
    assert Fraction(13)-8-4-Fraction(1, 2) == Fraction(1, 2)


def test_exact_interval_arithmetic_and_outward_rounding():
    a, b = Interval(-2, 3), Interval(-4, 5)
    assert a+b == Interval(-6, 8)
    assert a-b == Interval(-7, 7)
    assert a*b == Interval(-12, 15)
    assert a.square() == Interval(0, 9)
    assert Interval(-3, -2).square() == Interval(4, 9)
    exact = Interval(Fraction(-1, 3), Fraction(2, 3))
    rounded = outward(exact, bits=4)
    assert rounded == Interval(Fraction(-6, 16), Fraction(11, 16))
    assert rounded.lo <= exact.lo <= exact.hi <= rounded.hi
    with pytest.raises(TypeError):
        Interval(.1, .2)
    with pytest.raises(ValueError):
        Interval(1, 0)


def test_taylor_enclosures_at_known_values_and_nonzero_arguments():
    assert exp_point(0) == Interval(1, 1)
    assert sin_cos_point(0) == (Interval(0, 0), Interval(1, 1))
    e, (s, c) = exp_point(1), sin_cos_point(1)
    assert Fraction("2.718") < e.lo < e.hi < Fraction("2.719")
    assert Fraction(".841") < s.lo < s.hi < Fraction(".842")
    assert Fraction(".540") < c.lo < c.hi < Fraction(".541")
    # An interval test checks the Lipschitz extension as well as point Taylor sums.
    sine, cosine = sin_cos_interval(Interval(0, 1))
    assert sine.lo <= 0 and sine.hi >= s.hi
    assert cosine.lo <= c.lo and cosine.hi >= 1
    for fun, argument in [(exp_point, 5), (sin_cos_point, 17)]:
        with pytest.raises(ValueError):
            fun(argument)


@pytest.fixture(scope="module")
def proof():
    return global_minimum_proof()


def test_exact_cover_proves_three_roots_and_a_unique_lowest_minimum(proof):
    checked = verify_cover(proof["cells"])
    assert len(checked) == len(proof["roots"]) == 3
    assert [sign for _, sign in checked] == [1, -1, 1]
    roots = proof["roots"]
    assert sum(r["is_global_minimum"] for r in roots) == 1
    for (bracket, sign), row in zip(checked, roots):
        assert bracket.hi-bracket.lo <= Fraction(1, 2**40)
        assert derivative_bounds(Interval.point(bracket.lo))[0].sign == -sign
        assert derivative_bounds(Interval.point(bracket.hi))[0].sign == sign
        x = np.array([row["a"], -row["a"]])
        assert np.linalg.norm(model.gradient(x)) < 1e-8
        assert (np.linalg.eigvalsh(model.hessian(x)).min() > 0) == (sign > 0)
    # The global comparison is an EXACT rational ordering, not a float tolerance.
    global_row = proof["global_root"]
    other = next(r for r in roots if r["classification"] == "minimum" and not r["is_global_minimum"])
    assert Fraction(global_row["value_upper"]) < Fraction(other["value_lower"])
    assert Fraction(global_row["value_upper"]) < Fraction("-1.49914482")
    assert Fraction(global_row["value_lower"]) > Fraction("-1.49914483")
    assert Fraction(other["value_lower"]) > Fraction("-.55992883")


def test_rigorous_root_brackets_agree_with_original_numerical_search(proof):
    from sc_cw1.exercise1.part_c_minima import multistart
    _, _, points = multistart(5)
    for point in points:
        a = min(proof["roots"], key=lambda r: abs(r["a"]-point["x1"]))["a"]
        assert_allclose([point["x1"], point["x2"]], [a, -a], atol=1e-5)


def test_proof_fails_closed_on_exhaustion_or_corrupt_coverage(proof):
    with pytest.raises(ProofFailure):
        global_minimum_proof(max_depth=0)
    with pytest.raises(ProofFailure):
        verify_cover(proof["cells"][:-1])
    corrupted = copy.deepcopy(proof["cells"])
    corrupted[0]["derivative_lower"] = "0"
    with pytest.raises(ProofFailure):
        verify_cover(corrupted)


def test_exported_proof_cells_can_be_rechecked_after_csv_roundtrip(proof):
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(proof["cells"][0]))
    writer.writeheader()
    writer.writerows(proof["cells"])
    stream.seek(0)
    assert verify_cover(list(csv.DictReader(stream))) == verify_cover(proof["cells"])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_reduced_model_rejects_nonfinite_inputs(value):
    for fun in (reduced_objective, reduced_derivative, reduced_second_derivative):
        with pytest.raises(FloatingPointError):
            fun(value)
