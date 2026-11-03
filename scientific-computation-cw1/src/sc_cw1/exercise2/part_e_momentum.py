"""2(e): enable the printed momentum term in the ordinary descent loop."""

from .part_d_gradient_descent import gradient_descent
from .model import F, jacobian
from sc_cw1.common.utilities import module_cli


def momentum_descent(x0=(0., -1.), gamma=.3, step_size=.05, tol=1e-6, maxit=200,
                     residual=F, jac=jacobian):
    """Use d_k=gamma*d_(k-1)+2J.T F with zero initial direction."""
    if not 0 < gamma < 1:
        raise ValueError("momentum gamma must be between 0 and 1")
    return gradient_descent(x0=x0, step_size=step_size, tol=tol, maxit=maxit,
                            gamma=gamma, residual=residual, jac=jac)


def run(output="results"):
    """Demonstrate 2(e); 2(f) supplies the required independent comparison runs."""
    result = momentum_descent(gamma=.3)
    print(result["status"], result["iterations"], result["x"])
    return result


if __name__ == "__main__":
    module_cli(run)
