# Exercise 1(c): global-minimum analysis after Review 1

The reviewer proposed a dimension reduction. Every algebraic step is verified
below against the original objective in `exercise1/model.py`. The reduction and
tail bounds are analytical. A separate **exact rational interval calculation**
then proves a complete stationary-point count and a strict objective ordering.
This is a rigorous computer-assisted argument, not a dense-grid inference or a
claim that a floating-point iterate is an exact minimizer. It leaves the
coursework's Newton/BFGS implementations and experiments unchanged.

## 1. Verify the coordinate change

Define the orthogonal transformation

\[
u=(x_1-x_2)/\sqrt2,\quad v=(x_1+x_2)/\sqrt2,
\qquad x_1=(u+v)/\sqrt2,\quad x_2=(v-u)/\sqrt2.
\]

Direct expansion gives

\[
x_1^2+x_2^2=u^2+v^2,\qquad 2x_1x_2=v^2-u^2,
\]

and, importantly, the shifted product is

\[
\begin{aligned}
2(x_1-\tfrac12)(x_2+\tfrac12)
&=2x_1x_2+x_1-x_2-\tfrac12\\
&=v^2-u^2+\sqrt2u-\tfrac12.
\end{aligned}
\]

Therefore the reviewer's transformed expression is correct:

\[
\widetilde J(u,v)=e^{(u^2+v^2)/2}
 +\sin^2(v^2-u^2)+v^2-u^2+\sqrt2u-\tfrac12.
\]

For fixed u, put t=v²≥0 and write this expression as G(u,t). Its derivative is

\[
G_t=\tfrac12e^{(u^2+t)/2}+\sin(2(t-u^2))+1
\ge\tfrac12e^{(u^2+t)/2}>0.
\]

Here d(sin²z)/dz=sin(2z); this is why there is no extra factor of two
in the sine term of G_t. Thus G is strictly increasing in t for every fixed u.
Replacing any nonzero v by zero strictly lowers the objective. Consequently
**every global minimizer satisfies v=0**, equivalently x2=-x1.

In fact every stationary point also has v=0, since
\(\partial_v\widetilde J=2vG_t\) and G_t is strictly positive. This stronger
observation will connect the one-dimensional root count to the two-dimensional
stationary-point classifications.

## 2. Reduced objective and analytical derivatives

On v=0 set a=x1=-x2, so u=√2a. The even sine square removes the sign of
its argument and the reduced objective is exactly

\[
j(a)=J(a,-a)=e^{a^2}+\sin^2(2a^2)-2a^2+2a-\tfrac12.
\]

Differentiating gives

\[
\boxed{j'(a)=2ae^{a^2}+4a\sin(4a^2)-4a+2,}
\]

\[
\boxed{j''(a)=(2+4a^2)e^{a^2}+4\sin(4a^2)
                  +32a^2\cos(4a^2)-4.}
\]

The chain-rule factor in the second trigonometric term is
\(4a\cdot8a\cos(4a^2)=32a^2\cos(4a^2)\).
Independent checks compare these formulas with central differences and with
\(j'=(1,-1)\nabla J\),
\(j''=(1,-1)H(1,-1)^T\). The floating-point checks are regressions;
they are not the proof of the global claim.

## 3. Existence and an explicit finite global-search interval

Let s=|a|. Since sin²≥0,

\[
j(a)\ge b(s):=e^{s^2}-2s^2-2s-\tfrac12.
\]

The bound e^(s²)≥s⁶/6 from its positive power series shows b(s)→∞ as
s→∞. Thus j is continuous and coercive, and has a global minimizer.
Together with the strict v² reduction, this also proves existence of a
global minimizer of the original two-dimensional problem.

For s≥2, the positive exponential series gives e^(s²)≥e⁴>1+4+8=13.
Hence

\[
b'(s)=2s(e^{s^2}-2)-2>22s-2>0,
\quad b(2)=e^4-12.5>0.5=j(0).
\]

No global minimizer can have |a|≥2. Moreover, for a>0,

\[
j(a)-j(-a)=4a>0,
\]

so a positive point cannot be globally optimal. Since j'(0)=2, zero is not
a minimizer either. **Every possible global minimizer lies in (-2,0)**;
the explicit closed interval [-2,0] therefore contains them all.

## 4. Why all stationary points can be checked on [-2,2]

For completeness beyond the global candidates, write

\[
j'(a)=2aK(a)+2,\qquad
K(a)=e^{a^2}-2+2\sin(4a^2)\ge e^{a^2}-4.
\]

For |a|≥2, K(a)>9. Therefore j'(a)>0 when a≥2, while for a≤-2,
j'(a)<18a+2≤-34<0. There are **no stationary points outside [-2,2]**,
nor at its endpoints. It is sufficient to isolate every zero on this interval.
Positive a were excluded as *global minimizers* in section 3, not silently
excluded from this stronger stationary-point calculation.

## 5. Rigorous root count using exact rational bounds

`src/sc_cw1/exercise1/global_analysis.py` implements the following check with
standard-library `fractions.Fraction`. Every sign, coverage check and objective
ordering uses rational arithmetic. NumPy floats appear only in the separate
coordinate/derivative reporting functions, never in a proof decision.

For each rational interval A in [-2,2], compute enclosing intervals for j'(A)
and j''(A). Only these elementary enclosures are needed:

1. **Exponential, 0≤q≤4.** Sum S40=Σ(k=0..40)q^k/k! exactly. The remaining
   positive terms have successive ratios at most q/42<1, so
   \[
   S_{40}\le e^q\le S_{40}+
        \frac{q^{41}/41!}{1-q/42}.
   \]
   Monotonicity extends point bounds to an interval of q values.
2. **Sine/cosine, |q|≤16.** Use their degree-95 Taylor polynomials about zero
   (the cosine's degree-95 coefficient is zero). Taylor's remainder is bounded
   by |q|⁹⁶/96!, since all sine/cosine derivatives have absolute value at most
   one. These are rational bounds. For an argument interval with midpoint m
   and radius r, enlarge the bounds at m by r using |sin'|,|cos'|≤1, then
   intersect with [-1,1]. No approximate π or range-reduction assumption is used.
3. **Outward rounding.** Elementary point enclosures are rounded *outwards*
   to multiples of 2^-60 using integer floor/ceiling, solely to keep rational
   sizes modest. Subsequent interval operations use exact rational arithmetic.

Each final cell must satisfy one of these proofs:

- j'(A) excludes zero: no root in the cell.
- j''(A) has a strict sign, and the two endpoint derivatives have the same
  strict sign: monotonicity excludes a root.
- j''(A) has a strict sign, and endpoint derivatives have opposite strict
  signs: continuity gives a root and strict monotonicity gives **exactly one**.

Otherwise the interval is bisected (at most 20 covering-subdivision levels). A
depth limit or an undecidable sign raises
`ProofFailure`; it never becomes evidence of completeness. Each unique root is
further enclosed by sign-validated bisection in a bracket of width at most
2^-40. A second pass recomputes every bound, checks every reason and root
bracket, and checks that the final cells cover [-2,2] without gaps/overlaps.

The delivered run uses **40 covering cells**, with exactly **three unique-root
cells** and all other cells proven root-free. All cell endpoints, derivative
and curvature enclosures, endpoint signs and final root brackets are written to
`results/tables/ex1_part_c_interval_cover.csv`; they are not rounded decimal
sampling data. Reproduction is part of `python scripts/run_all.py`.

## 6. Classification and comparison

Let D=2⁴⁰=1099511627776. The exact sign-changing isolating intervals are:

| Root | Exact a interval (both integer endpoints divided by D) | Numerical a | Reduced type | Two-dimensional type |
| --- | --- | --- | --- | --- |
| 1 | [-1303253958365, -1303253958364]/D | -1.185302570197 | minimum | minimum |
| 2 | [-816195374817, -816195374816]/D | -0.742325369007 | maximum | saddle |
| 3 | [-725089608042, -725089608041]/D | -0.659465156824 | minimum | minimum |

At v=0, mixed u/v curvature is zero because G(u,v²) is even in v. The Hessian
in the orthogonal (u,v) coordinates has diagonal entries

\[
\partial_{uu}\widetilde J=\tfrac12j''(a),\qquad
\partial_{vv}\widetilde J=e^{a^2}+2-2\sin(4a^2)\ge e^{a^2}>0.
\]

The root cells prove respectively j''>0, j''<0, j''>0 throughout. Thus the
reduced maximum is a saddle in two dimensions; the two reduced minima are
strict two-dimensional minima. The coverage and tail proof establish exactly
these three stationary points in the plane, rather than inferring completeness
from the 441 starts.

Interval substitution into j encloses each exact stationary value. For readable
presentation, the following decimal endpoints are deliberately widened rational
outer bounds; full rational enclosures are in
`results/tables/ex1_part_c_reduced_stationary_points.csv`:

| Root | Proven enclosing objective interval | Numerical objective |
| --- | --- | --- |
| 1 | (-1.49914483, -1.49914482) | -1.499144825603 |
| 2 | (-0.55573516, -0.55573515) | -0.555735151910 |
| 3 | (-0.55992883, -0.55992882) | -0.559928823757 |

The upper bound for minimum 1 is strictly less than the lower bound for
minimum 3. Existence, the dimension reduction, complete root isolation and
this strict ordering together prove that **the original objective has a unique
true global minimizer (a_g,-a_g)**, where a_g is the unique zero of j' in root
interval 1. Its numerical coordinates are approximately

\[
(-1.1853025702,\;1.1853025702),\qquad J\approx-1.4991448256.
\]

These decimal coordinates are not claimed to be exact, nor does the numerical
Newton tolerance itself prove global optimality.

## 7. Limits, source attribution and reporting

- This is a computer-assisted proof based on explicitly justified rational
  enclosures, not a formally verified theorem-prover development. Independent
  review should check the reduction, elementary remainder formulas, interval
  code and cover data. Increasing a floating-point grid would not replace them.
- The reviewer supplied the coordinate-reduction idea. The expansion,
  derivatives, analytical bounds and interval argument here were independently
  checked and implemented for Review 1. They are supplementary analysis of the
  original brief's equation, not new requirements attributed to the lecturer.
- The proof invokes elementary continuity/monotonicity, existence of minima
  on compact sets and Taylor remainder bounds. The separate report should
  explain or reference these where required by the original referencing rules.
- The exact brackets/objective bounds support the claim. The original numerical
  search/history tables remain unchanged; for example the multistart saddle
  coordinates differ from the tighter supplementary approximation within their
  original stopping accuracy. Those rows must not be replaced by refined answers.
- No claim of exhaustive Exercise 2 root/minimum counting follows from this
  Exercise 1 proof. That exercise's apparent/graphical count remains numerical.
