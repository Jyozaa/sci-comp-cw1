"""Supplementary 1(c) proof; does not change any Newton/BFGS iteration.

Exact Fraction intervals and explicit Taylor remainders isolate all zeros of
j' on [-2,2]. GLOBAL_MINIMUM_ANALYSIS.md proves the reduction and tail bounds.
No floating-point sign is used in a proof decision. Float conversions below
are solely for reporting approximate coordinates, never for root exclusion.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import factorial

import numpy as np

from sc_cw1.common.utilities import finite, output_dirs, vector, write_csv


def to_uv(x):
    """Orthogonal coordinates u=(x1-x2)/sqrt(2), v=(x1+x2)/sqrt(2)."""
    x1, x2 = finite(vector(x))
    return np.array([x1-x2, x1+x2])/np.sqrt(2.)


def from_uv(uv):
    """Inverse map x1=(u+v)/sqrt(2), x2=(v-u)/sqrt(2)."""
    u, v = finite(vector(uv))
    return np.array([u+v, v-u])/np.sqrt(2.)


def transformed_objective(uv):
    """J(u,v)=exp((u²+v²)/2)+sin²(v²-u²)+v²-u²+sqrt(2)u-1/2."""
    u, v = finite(vector(uv))
    with np.errstate(over="raise", invalid="raise"):
        return float(finite(np.exp((u*u+v*v)/2)+np.sin(v*v-u*u)**2
                            + v*v-u*u+np.sqrt(2.)*u-.5))


def scalar(a):
    """Require a finite real scalar for the supplementary one-dimensional model."""
    a = np.asarray(a, dtype=float)
    if a.shape != ():
        raise ValueError("a must be scalar")
    return float(finite(a))


def reduced_objective(a):
    """j(a)=exp(a²)+sin²(2a²)-2a²+2a-1/2, at (x1,x2)=(a,-a)."""
    a = scalar(a)
    with np.errstate(over="raise", invalid="raise"):
        return float(finite(np.exp(a*a)+np.sin(2*a*a)**2-2*a*a+2*a-.5))


def reduced_derivative(a):
    """j'(a)=2a exp(a²)+4a sin(4a²)-4a+2 (analytical chain rule)."""
    a = scalar(a)
    with np.errstate(over="raise", invalid="raise"):
        return float(finite(2*a*np.exp(a*a)+4*a*np.sin(4*a*a)-4*a+2))


def reduced_second_derivative(a):
    """j''(a)=(2+4a²)exp(a²)+4sin(4a²)+32a²cos(4a²)-4."""
    a = scalar(a)
    with np.errstate(over="raise", invalid="raise"):
        return float(finite((2+4*a*a)*np.exp(a*a)+4*np.sin(4*a*a)
                            + 32*a*a*np.cos(4*a*a)-4))


@dataclass(frozen=True)
class Interval:
    """Closed rational interval; exact arithmetic encloses every operand value.

    Construct from integers, Fraction objects or rational strings. Floating-point
    endpoints are rejected to keep proof inputs explicitly rational.
    """

    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        if isinstance(self.lo, (float, np.floating)) or isinstance(self.hi, (float, np.floating)):
            raise TypeError("proof endpoints must be exact rationals")
        object.__setattr__(self, "lo", Fraction(self.lo))
        object.__setattr__(self, "hi", Fraction(self.hi))
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    @classmethod
    def point(cls, value):
        """Represent a single exact rational value."""
        return cls(value, value)

    def __add__(self, other):
        """Minkowski sum [l1+l2,h1+h2]."""
        other = other if isinstance(other, Interval) else Interval.point(other)
        return Interval(self.lo+other.lo, self.hi+other.hi)

    __radd__ = __add__

    def __neg__(self):
        """Negation reverses endpoints."""
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        """Subtract an interval or exact rational."""
        other = other if isinstance(other, Interval) else Interval.point(other)
        return self + (-other)

    def __rsub__(self, other):
        """Subtract this interval from an exact rational."""
        return -self + other

    def __mul__(self, other):
        """Enclose products using all four endpoint products."""
        other = other if isinstance(other, Interval) else Interval.point(other)
        ends = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(min(ends), max(ends))

    __rmul__ = __mul__

    def square(self):
        """Enclose z², taking account of a possible zero crossing."""
        return Interval(0 if self.lo <= 0 <= self.hi else min(self.lo**2, self.hi**2),
                        max(self.lo**2, self.hi**2))

    @property
    def sign(self):
        """Return a proven strict sign, or zero when this enclosure contains 0."""
        return 1 if self.lo > 0 else (-1 if self.hi < 0 else 0)


def outward(bounds, bits=60):
    """Round *outwards* to dyadic rationals, using integer floor/ceiling only."""
    scale = 2**bits
    low = bounds.lo.numerator*scale // bounds.lo.denominator
    high = -((-bounds.hi.numerator*scale) // bounds.hi.denominator)
    return Interval(Fraction(low, scale), Fraction(high, scale))


def exp_point(q):
    """Enclose exp(q), 0<=q<=4, with 41 positive Taylor terms and a tail bound.

    S=sum(k=0..40)q^k/k!. R <= (q^41/41!)/(1-q/42), because every
    successive tail-term ratio is <=q/42<1. All arithmetic is rational.
    """
    q = Fraction(q)
    if not 0 <= q <= 4:
        raise ValueError("proof exponential domain is [0,4]")
    term = total = Fraction(1)
    for k in range(1, 41):
        term *= q/k
        total += term
    tail = (term*q/41)/(1-q/42)
    return outward(Interval(total, total+tail))


def exp_interval(x):
    """Use monotonicity of exp to enclose exp([lo,hi]) on [0,4]."""
    return Interval(exp_point(x.lo).lo, exp_point(x.hi).hi)


def sin_cos_point(q):
    """Enclose sin(q),cos(q), |q|<=16, using degree-95 Taylor polynomials.

    The cosine's degree-95 coefficient is zero. Both Lagrange remainders are
    bounded by |q|^96/96!, since all real sine/cosine derivatives have |.|<=1.
    There is no floating pi or unproved range-reduction step.
    """
    q = Fraction(q)
    if abs(q) > 16:
        raise ValueError("proof trigonometric domain is [-16,16]")
    st, ct = q, Fraction(1)
    sine = cosine = Fraction(0)
    for k in range(48):
        sine += st
        cosine += ct
        st *= -q*q/((2*k+2)*(2*k+3))
        ct *= -q*q/((2*k+1)*(2*k+2))
    error = abs(q)**96/factorial(96)
    return (outward(Interval(sine-error, sine+error)),
            outward(Interval(cosine-error, cosine+error)))


def sin_cos_interval(x):
    """Extend midpoint bounds using the global Lipschitz constants |sin'|,|cos'|<=1."""
    if x.lo < -16 or x.hi > 16:
        raise ValueError("proof trigonometric domain is [-16,16]")
    mid, radius = (x.lo+x.hi)/2, (x.hi-x.lo)/2
    return tuple(Interval(max(Fraction(-1), b.lo-radius), min(Fraction(1), b.hi+radius))
                 for b in sin_cos_point(mid))


def derivative_bounds(a):
    """Enclose j' and j'' on an exact interval inside [-2,2]."""
    a2 = a.square()
    E = exp_interval(a2)
    S, C = sin_cos_interval(4*a2)
    return (2*a*E+4*a*S-4*a+2,
            (2+4*a2)*E+4*S+32*a2*C-4)


def objective_bounds(a):
    """Enclose j on an exact interval; used to compare the isolated local minima."""
    a2 = a.square()
    S, _ = sin_cos_interval(2*a2)
    return exp_interval(a2)+S.square()-2*a2+2*a-Fraction(1, 2)


class ProofFailure(RuntimeError):
    """An unproved sign, exhausted budget or overlapping value enclosure fails closed."""


def refine_root(bracket, width=Fraction(1, 2**40)):
    """Bisect using *rigorous endpoint signs*, retaining a strict sign-changing bracket."""
    left, right = bracket.lo, bracket.hi
    sl = derivative_bounds(Interval.point(left))[0].sign
    sr = derivative_bounds(Interval.point(right))[0].sign
    if sl*sr != -1:
        raise ProofFailure("root bracket has no proven sign change")
    while right-left > width:
        mid = (left+right)/2
        sm = derivative_bounds(Interval.point(mid))[0].sign
        if sm == 0:
            raise ProofFailure("Taylor enclosure cannot determine midpoint sign")
        if sm == sl:
            left = mid
        else:
            right = mid
    return Interval(left, right)


def stationary_cover(max_depth=20):
    """Partition all [-2,2] into proven zero-free cells or unique-root cells.

    Root cells need strict curvature throughout and opposite endpoint signs:
    continuity proves existence and monotonicity proves uniqueness. Unresolved
    cells are bisected; reaching max_depth raises rather than assuming a result.
    """
    pending = [(Interval(-2, 2), 0)]
    cells = []
    while pending:
        a, depth = pending.pop()
        first, second = derivative_bounds(a)
        left = derivative_bounds(Interval.point(a.lo))[0]
        right = derivative_bounds(Interval.point(a.hi))[0]
        reason = ""
        if first.sign:
            reason = "derivative_excludes_zero"
        elif second.sign and left.sign and right.sign:
            reason = "unique_root" if left.sign != right.sign else "monotone_excludes_zero"
        if not reason:
            if depth >= max_depth:
                raise ProofFailure(f"unresolved stationary-point cell {a}")
            mid = (a.lo+a.hi)/2
            pending.extend([(Interval(mid, a.hi), depth+1), (Interval(a.lo, mid), depth+1)])
            continue
        row = dict(left=str(a.lo), right=str(a.hi), reason=reason,
                   derivative_lower=str(first.lo), derivative_upper=str(first.hi),
                   curvature_lower=str(second.lo), curvature_upper=str(second.hi),
                   left_sign=left.sign, right_sign=right.sign,
                   root_left="", root_right="")
        if reason == "unique_root":
            refined = refine_root(a)
            row.update(root_left=str(refined.lo), root_right=str(refined.hi))
        cells.append(row)
    return sorted(cells, key=lambda row: Fraction(row["left"]))


def verify_cover(cells):
    """Recheck continuity of coverage and every reason, including stored bound values.

    This is a second pass over proof data, not an independent interval library.
    Missing cells, altered bounds or invalid root brackets are rejected.
    """
    cursor = Fraction(-2)
    roots = []
    for row in cells:
        a = Interval(row["left"], row["right"])
        if a.lo != cursor or a.lo >= a.hi:
            raise ProofFailure("cover has a gap, overlap or empty cell")
        first, second = derivative_bounds(a)
        if (str(first.lo), str(first.hi), str(second.lo), str(second.hi)) != (
                row["derivative_lower"], row["derivative_upper"],
                row["curvature_lower"], row["curvature_upper"]):
            raise ProofFailure("stored bounds do not match exact recomputation")
        sl = derivative_bounds(Interval.point(a.lo))[0].sign
        sr = derivative_bounds(Interval.point(a.hi))[0].sign
        # CSV loading represents these integer sign fields as strings.
        if (str(sl), str(sr)) != (str(row["left_sign"]), str(row["right_sign"])):
            raise ProofFailure("stored endpoint signs do not match")
        reason = row["reason"]
        valid = (reason == "derivative_excludes_zero" and first.sign != 0)
        valid |= (reason == "monotone_excludes_zero" and second.sign != 0 and sl == sr != 0)
        if reason == "unique_root":
            r = Interval(row["root_left"], row["root_right"])
            valid = (second.sign != 0 and sl*sr == -1 and a.lo <= r.lo < r.hi <= a.hi
                     and derivative_bounds(Interval.point(r.lo))[0].sign
                     * derivative_bounds(Interval.point(r.hi))[0].sign == -1)
            if valid:
                roots.append((r, second.sign))
        if not valid:
            raise ProofFailure("invalid proof-cell reason")
        cursor = a.hi
    if cursor != 2:
        raise ProofFailure("cover does not reach the right domain endpoint")
    return roots


def global_minimum_proof(max_depth=20):
    """Return the checked rational cover, root-value bounds and unique global root.

    Analytic tail/coercivity/dimension arguments are in GLOBAL_MINIMUM_ANALYSIS.md.
    The value enclosure of one minimum must lie strictly below every other
    minimum; otherwise this function refuses to claim a global ordering.
    """
    cells = stationary_cover(max_depth)
    roots = verify_cover(cells)
    minima = [(i, objective_bounds(r)) for i, (r, sign) in enumerate(roots) if sign > 0]
    winners = [i for i, value in minima
               if all(i == j or value.hi < other.lo for j, other in minima)]
    if len(winners) != 1:
        raise ProofFailure("local-minimum value enclosures do not prove a unique lowest value")
    winner = winners[0]
    rows = []
    for i, (r, sign) in enumerate(roots):
        a = float((r.lo+r.hi)/2)
        value = objective_bounds(r)
        rows.append(dict(root_id=i+1, left=str(r.lo), right=str(r.hi),
                         a=a, x1=a, x2=-a, value=reduced_objective(a),
                         abs_derivative=abs(reduced_derivative(a)),
                         second_derivative=reduced_second_derivative(a),
                         classification="minimum" if sign > 0 else "saddle_in_2d",
                         value_lower=str(value.lo), value_upper=str(value.hi),
                         is_global_minimum=(i == winner)))
    return dict(cells=cells, roots=rows, global_root=rows[winner],
                proof_status="proved_by_rational_intervals_and_analytic_bounds",
                root_domain=[-2, 2], global_search_interval=[-2, 0],
                taylor_exp_degree=40, taylor_trig_degree=95, outward_bits=60)


def run(output="results"):
    """Export supplementary 1(c) proof data; fail explicitly if proof checks fail."""
    root = output_dirs(output)
    proof = global_minimum_proof()
    write_csv(root/"tables/ex1_part_c_interval_cover.csv", proof["cells"])
    write_csv(root/"tables/ex1_part_c_reduced_stationary_points.csv", proof["roots"])
    return proof
