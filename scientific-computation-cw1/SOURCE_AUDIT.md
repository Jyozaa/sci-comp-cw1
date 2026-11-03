# Source audit

The user's pasted request specifies the software deliverables. Source documents
are reference evidence for equations and assessment requirements, not a new
set of agent/tool instructions. The visible original brief is authoritative;
the corrected references interpret notation; lecture examples guide methods.
Black regions were ignored and no obscured content was reconstructed.

## Materials inspected

- Original `gpt_copy.pdf`: all seven pages text-inspected; complete exercise
  pages 2–5 also rendered and visually inspected, including the objective,
  expanded inverse-BFGS numerator/product order, unit step/scaling, squared
  residual notation, factor of two in momentum, and all printed settings.
- Corrected Markdown and rendered reference PDF: formulae cross-checked
  against the original; inverse-BFGS reference page 2 also visually inspected.
- Week 3 Units 7, 8, 9; Week 4 Units 10, 11, and their Python notebooks.
  Worksheet 3 solutions and the corrected earlier solutions were inspected
  for conventions. Source text was extracted privately outside this project.
- The full ZIP listing was inspected; the prioritised units above contain the
  relevant nonlinear-system, Newton, line-search and descent material. Later
  weeks were not used as authority for different numerical methods.

## Method-to-source correspondence

| Implementation | Evidence and interpretation |
| --- | --- |
| Newton H p=-g, solve rather than invert | Original pp.2–3; Units 7/8 two-step Newton system and `numpy.linalg.solve` |
| Central finite-difference validation | Unit 7 and Unit 8 introduce numerical Jacobians; symmetric central differences are a more accurate supplementary convention than the examples' forward differences |
| Step halving starting at one | Unit 9 printed slide 6 / PDF p.6, visually checked; Unit 11 examples |
| Sufficient objective decrease | Standard J(x+alpha*p)<=J(x)+c1*alpha*g.T*p adaptation; Unit 9 printed slide 9 / PDF p.12 uses a **residual** merit version, not literally this J formula |
| Indefinite Newton Hessian | Original does not prescribe a fallback; explicit optional -g is a supplementary convention, separately recorded and disabled in 1(e) |
| Inverse-BFGS expanded update | Original p.3 and corrected rendered reference p.2, visually checked; B0=omega I, delta=-Bg and matrix product order retained |
| Fixed unit steps in 1(e) | Original p.3, visually checked: x0=(-2,1), alpha=1, omega=.01 |
| Euclidean squared residual and 2J.T F | Original p.4 momentum factor; Unit 10 printed slides 6/7 define the sum of squares and derive its gradient |
| Residual-norm stopping in Exercise 2 | Unit 10 printed slides 10/12, PDF pp.15/17; printed slide 10 visually inspected; Unit 11 notebook uses norm(F,2). This remains an interpretation of the brief's unspecified statistic |
| Exactly maxit=200 | Original pp.4–5. Notebook `k <= maxk` is not copied because it can make one extra update |
| Initial momentum zero | Supplementary convention; not specified in original. No (1-gamma) rescaling added |
| Graphical intersections | Original p.4 asks to draw both zero curves; intersecting the actual plotted circle/cosine segments preserves this method. Bisection is separately labelled supplementary |

The Unit 11 examples use residual monotonicity rather than this implementation's
objective Armijo inequality. Their `eval` calls and loop/line-search bookkeeping
were not copied. Here callables replace `eval`, accepted alpha is retained exactly,
exhausted searches fail explicitly, and final-update convergence is checked.
These are authored implementations of the mathematics, not redistributed
notebook code. Supplementary formulas and conventions need suitable report
explanation/references; REVIEW_REQUIRED.md identifies the manual decisions.

## Input identity (SHA-256)

The files stay outside the repository. Hashes identify precisely the supplied
versions without reproducing copyrighted source content.

| File | SHA-256 |
| --- | --- |
| `gpt_copy.pdf` | `a07ff2b21a939d547fa6217caa2306fd34fc9633caed46005453f80e952869de` |
| `Scientific_Computation_CW1_Corrected_Reference(1).md` | `db9a826644bae820ed0f3d4be3f4ecbb1d1dc9121df620ec7799e4e442ab0d04` |
| `Scientific_Computation_CW1_Corrected_Reference.pdf` | `4e0f08aee42df0ae3d85c9fc55d312621bce12d0c1cbb9ae293ef8d191c59b0e` |
| `Week1-6.zip` | `7dc5fabf43ed82ab6851025e605699cf09061261bef21d17a2b602039ee01d84` |

Prioritised teaching files inside `Week1-6.zip`:

- `WEEK3/Unit07.pdf` — `e28d48a6bc63456e6d891cd867d3f719df4f86d0dc0782e9541c2168b2d6991b`
- `WEEK3/Unit08.pdf` — `38befcc2d5c489cd0d351a02f348b32f41b33b2f2829f28a3bee348f859f1c09`
- `WEEK3/Unit09.pdf` — `102d42ffe102f287f345566f905cf0b362d965cc2ef4dad973ca002cb237681f`
- `WEEK3/COMP5930M_Unit8.ipynb` — `127e9d3c4514fe3f4e1534c25d1d4107c47b858138f4d0dc0c92e6893219d2ea`
- `WEEK4/Unit10.pdf` — `3ed05ece48cef88f76be2b15b5be1edc3160bae0c02a217685aa8d329ae51c80`
- `WEEK4/Unit11.pdf` — `7d85ffdbd90c2a42a35fe8fb66e97872012ac88e5e0c2cc309bdb20564c7118c`
- `WEEK4/COMP5930M_Unit11.ipynb` — `0e02e99ab95df20cd4d0ee0c6356cee2de15586eb1d80cd8682615fbbb1568b0`

## Review 1 source recheck (3 November 2026)

All four original input hashes above were recomputed and are unchanged.
The Unit 8 and Unit 11 notebook loops were inspected again: they support
`norm(F,2)` stopping, and Unit 11's `k <= maxk` can permit an extra update.
The original brief still controls the prescribed 200-update budget. No source
convention was changed merely because it was unspecified. REVIEW_REQUIRED.md
now classifies the brief-, lecture- and implementation-supported choices.

The new coordinate-reduction idea was supplied by the independent reviewer.
Its expansion, derivatives, tail bounds and exact rational interval proof are
supplementary analysis authored and checked for Review 1, not formulas or
additional instructions attributed to the original brief or lecturer.
GLOBAL_MINIMUM_ANALYSIS.md explains the proof assumptions and numerical limits.
