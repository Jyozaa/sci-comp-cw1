"""2(a): nonlinear residuals, analytical Jacobian and squared residual."""

import numpy as np


def F(x):
    """F1=(u-1)²+v²-1; F2=cos(2u)-v."""
    u, v = np.asarray(x, dtype=float)
    return np.array([(u-1)**2 + v**2 - 1, np.cos(2*u) - v])


def jacobian(x):
    """J_ij=∂F_i/∂x_j."""
    u, v = np.asarray(x, dtype=float)
    return np.array([[2*(u-1), 2*v], [-2*np.sin(2*u), -1.]])


def phi(x):
    """phi=||F||₂²=F1²+F2²."""
    residual = F(x)
    return float(residual @ residual)


def grad_phi(x):
    """The gradient of the squared residual is 2J.T F."""
    return 2*jacobian(x).T @ F(x)


def residual_grid(X, Y):
    """Evaluate the same residual equations on a meshgrid."""
    return (X-1)**2 + Y**2 - 1, np.cos(2*X) - Y


def phi_grid(X, Y):
    """Evaluate F1²+F2² on a meshgrid."""
    f1, f2 = residual_grid(X, Y)
    return f1**2 + f2**2
