"""1(a): the objective and its analytical derivatives (angles in radians)."""

import numpy as np


def objective(x):
    """J=exp((u²+v²)/2)+sin²(2uv)+2(u-1/2)(v+1/2)."""
    u, v = np.asarray(x, dtype=float)
    return float(np.exp((u*u + v*v) / 2) + np.sin(2*u*v)**2 + 2*(u-.5)*(v+.5))


def gradient(x):
    """The first-order conditions set both components of this gradient to zero."""
    u, v = np.asarray(x, dtype=float)
    E, S = np.exp((u*u + v*v)/2), np.sin(4*u*v)
    # d(sin²(t))/dt=sin(2t), with t=2uv.
    return np.array([u*E + 2*v*S + 2*v + 1, v*E + 2*u*S + 2*u - 1])


def hessian(x):
    """Second derivatives of J; H12=H21 by symmetry."""
    u, v = np.asarray(x, dtype=float)
    E, C = np.exp((u*u + v*v)/2), np.cos(4*u*v)
    off = u*v*E + 2*np.sin(4*u*v) + 8*u*v*C + 2
    return np.array([[(1+u*u)*E+8*v*v*C, off], [off, (1+v*v)*E+8*u*u*C]])


def numerical_hessian(x, h=1e-5):
    """Check H by central differences of the analytical gradient."""
    x = np.asarray(x, dtype=float)
    H = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        H[:, j] = (gradient(x+e)-gradient(x-e))/(2*h)
    return H


def objective_grid(X, Y):
    """The same objective evaluated on meshgrid arrays."""
    return np.exp((X**2+Y**2)/2) + np.sin(2*X*Y)**2 + 2*(X-.5)*(Y+.5)
