"""1(a): objective and hand-derived first/second derivatives (angles in radians)."""

import numpy as np
from sc_cw1.common.utilities import vector, finite_difference_jacobian


def objective(x):
    """J=exp((u²+v²)/2)+sin²(2uv)+2(u-1/2)(v+1/2), scalar at x=(u,v)."""
    u, v = vector(x)
    return float(np.exp((u*u + v*v) / 2) + np.sin(2*u*v)**2 + 2*(u-.5)*(v+.5))


def gradient(x):
    """Analytical ∇J=[uE+2v sin(4uv)+2v+1, vE+2u sin(4uv)+2u-1].

    Chain rule: d sin²(t)/dt=sin(2t); t=2uv, E=exp((u²+v²)/2).
    The first-order conditions set both of these components equal to zero.
    """
    u, v = vector(x)
    E, S = np.exp((u*u + v*v)/2), np.sin(4*u*v)
    return np.array([u*E + 2*v*S + 2*v + 1, v*E + 2*u*S + 2*u - 1])


def hessian(x):
    """Exact H: H11=(1+u²)E+8v²C; H22=(1+v²)E+8u²C;
    H12=H21=uvE+2 sin(4uv)+8uvC+2, with C=cos(4uv).
    """
    u, v = vector(x)
    E, C = np.exp((u*u + v*v)/2), np.cos(4*u*v)
    off = u*v*E + 2*np.sin(4*u*v) + 8*u*v*C + 2
    return np.array([[(1+u*u)*E+8*v*v*C, off], [off, (1+v*v)*E+8*u*u*C]])


def numerical_hessian(x, h=1e-5):
    """Central differences of ∇J; unsymmetrised to expose derivative errors."""
    return finite_difference_jacobian(gradient, x, h)


def objective_grid(X, Y):
    """Vectorised J on matching mesh arrays; same equation as objective()."""
    return np.exp((X**2+Y**2)/2) + np.sin(2*X*Y)**2 + 2*(X-.5)*(Y+.5)
