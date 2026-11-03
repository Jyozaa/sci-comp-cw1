"""2(e): momentum extension; the three changed lines are marked in fixed_descent."""

import numpy as np
from sc_cw1.common.utilities import output_dirs, save_history, summary, write_csv, module_cli
from .model import F, jacobian
from .part_d_gradient_descent import fixed_descent


def momentum_descent(x0=(0., -1.), *, gamma=.3, step_size=.05, tol=1e-6,
                     maxit=200, initial_direction=None, residual=F, jac=jacobian):
    """Use printed d_k=gamma*d_{k-1}+2J.T F, x_{k+1}=x_k-lambda*d_k.

    Zero initial direction is an explicit convention, overridable for review.
    The common loop keeps all convergence/recording logic identical to 2(d).
    gamma*previous adds a decaying memory of earlier gradients to the current
    gradient; it is not (1-gamma)*g and not Nesterov's look-ahead gradient.
    """
    if not np.isfinite(gamma) or not 0 < gamma < 1:
        raise ValueError("momentum gamma must be strictly between 0 and 1")
    return fixed_descent(x0, gamma=gamma, step_size=step_size, tol=tol,
                         maxit=maxit, initial_direction=initial_direction, residual=residual, jac=jac)


def run(output="results"):
    """Standalone 2(e) illustration at gamma=.3 (the required sweep is 2(f))."""
    root = output_dirs(output)
    result = momentum_descent(gamma=.3)
    save_history(root / "tables/ex2_part_e_momentum_history.csv", result)
    write_csv(root / "tables/ex2_part_e_summary.csv", [summary(result)])
    return result


if __name__ == "__main__":
    module_cli(run)
