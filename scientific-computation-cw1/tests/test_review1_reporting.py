"""Review 1 checks for literal report excerpts and important convention boundaries."""

from pathlib import Path
import re
import textwrap

import numpy as np
import pytest

from sc_cw1.exercise1.part_e_comparison import compare
from sc_cw1.exercise2.part_d_gradient_descent import gradient_descent
from sc_cw1.exercise2.part_e_momentum import momentum_descent


def test_report_snippets_are_literal_contiguous_source_excerpts():
    root = Path(__file__).resolve().parents[1]
    document = (root/"REPORT_SNIPPETS.md").read_text()
    blocks = re.findall(r"Source: `([^`]+)`[^\n]*\n\n```python\n(.*?)\n```", document, re.S)
    assert len(blocks) == document.count("```python") == 9
    for filename, code in blocks:
        lines = (root/filename).read_text().splitlines()
        n = len(code.splitlines())
        assert any(textwrap.dedent("\n".join(lines[i:i+n])) == code
                   for i in range(len(lines)-n+1)), filename


def test_unit_step_comparison_never_calls_armijo(monkeypatch):
    from sc_cw1.exercise1 import part_b_newton, part_d_bfgs
    def forbidden(*args, **kwargs):
        raise AssertionError("1(e) must not call Armijo, even if it would return alpha=1")
    monkeypatch.setattr(part_b_newton, "armijo", forbidden)
    monkeypatch.setattr(part_d_bfgs, "armijo", forbidden)
    assert all(result.converged for result in compare().values())


@pytest.mark.parametrize("solver", [gradient_descent, momentum_descent])
def test_exact_200_update_budget_when_residual_cannot_vanish(solver):
    result = solver(residual=lambda x: np.ones(2), jac=lambda x: np.zeros((2, 2)),
                    maxit=200, tol=1e-6)
    assert result.status == "max_iterations" and not result.converged
    assert result.nit == 200 and len(result.history) == 201
    assert result.history[-1]["k"] == 200
