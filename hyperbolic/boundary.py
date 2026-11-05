"""Exact rational decisions for the analytic H^3 isometry control only.

Float64 ball coordinates incur rounding. Re-evaluate ambiguous control edges
using the exact rational map of the stored Euclidean coordinates, center, a and
eps. This does not enlarge either neighborhood or alter the native H^2 metric.
"""
from fractions import Fraction as F


def control_edge(xi, xj, center, a, eps):
    xi, xj, center = [[F(float(v)) for v in row] for row in (xi,xj,center)]
    a, eps = F(float(a)), F(float(eps))
    euclidean = sum((u-v)**2 for u,v in zip(xi,xj)) <= eps**2
    def ball(x):
        v = [a*(u-c) for u,c in zip(x,center)]
        s = sum(u*u for u in v)
        return [2*u/(s+4) for u in v]+[s/(s+4)]
    p, q = ball(xi), ball(xj)
    chord2 = sum((u-v)**2 for u,v in zip(p,q))
    deficit = (1-sum(u*u for u in p))*(1-sum(v*v for v in q))
    # sinh(eps_H/2)^2 = (a*eps_E/2)^2; all terms are rational.
    hyperbolic = 4*chord2 <= a*a*eps*eps*deficit
    if euclidean != hyperbolic:
        raise AssertionError('Exact control edge identity failed')
    return euclidean, hyperbolic
