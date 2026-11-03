"""2(a): F, its analytical Jacobian J, and the Euclidean squared residual."""

import numpy as np
from sc_cw1.common.utilities import vector


def F(x):
    """Return [(u-1)^2+v^2-1, cos(2u)-v] at x=(u,v), angles in radians."""
    u, v = vector(x)
    return np.array([(u-1)**2 + v**2 - 1, np.cos(2*u) - v])


def jacobian(x):
    """J_ij=∂F_i/∂x_j: [[2(u-1),2v],[-2sin(2u),-1]]."""
    u, v = vector(x)
    return np.array([[2*(u-1), 2*v], [-2*np.sin(2*u), -1.]])


def phi(x):
    """φ=||F||_2²=F1²+F2²; the Euclidean interpretation matches Unit 10."""
    residual = F(x)
    return float(residual @ residual)


def grad_phi(x):
    """∇φ=2 J(x).T F(x), using the analytical Jacobian of Exercise 2."""
    return 2 * jacobian(x).T @ F(x)


def hessian_phi(x):
    """Supplementary exact residual Hessian for classifying minima/slow modes.

    ∇²φ=2 J.T J+2 F1∇²F1+2 F2∇²F2, where ∇²F1=2I and
    ∇²F2=diag(-4cos(2u),0). Not used in the gradient descent iteration.
    """
    u, _ = vector(x)
    f = F(x)
    J = jacobian(x)
    return 2*J.T@J + 4*f[0]*np.eye(2) + 2*f[1]*np.diag([-4*np.cos(2*u), 0.])


def residual_grid(X, Y):
    """Vectorised F1,F2 on mesh arrays; exactly the same equations as F()."""
    return (X-1)**2 + Y**2 - 1, np.cos(2*X) - Y


def phi_grid(X, Y):
    """Vectorised φ=F1²+F2² on mesh arrays."""
    f1, f2 = residual_grid(X, Y)
    return f1**2 + f2**2
