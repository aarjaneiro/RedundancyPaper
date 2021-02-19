#cython: language_level=3

"""
Do not use directly (use exchange.py)!

Local exchangeability helpers
https://github.com/trevorcampbell/localexch
esp. https://github.com/trevorcampbell/localexch/blob/master/examples/crashdata/crash_analysis.ipynb
"""
from warnings import warn
import numpy as np
cimport numpy as np

cpdef local_empirical_measure(float tau, float[:] X, float[:] T, np.ufunc f, b = None):
    """
    As in Theorem 8 of Cambell et al., 2020
    """
    if b is not None:
        b0 = b(tau, T)
    else:
        b0 = 2 * np.sqrt(f(tau, T))
    bb = np.sort(b0)
    bsum = 0.
    for j in range(T.shape[0]):
        bsum += bb[j]
        if bb[j] >= (0.5 + bsum) / (j + 1):
            bsum -= bb[j]
            mu = (0.5 + bsum) / j
            return 2 * np.maximum(mu - b0, 0.) # can give a list of results

def local_empirical_estimate(np.ufunc h, float tau, float[:] X, float[:] T, np.ufunc f):
    """
    With

    .. math::
        b(\\tau,t) = 2* \\sqrt{f( \\tau,t )},

    provides an estimate of local_empirical_measure based on data.
    """
    cdef float t
    b = lambda tau, t: 2 * np.sqrt(f(tau, t))
    xi = local_empirical_measure(tau, X, T, f, b)
    return (xi * h(X)).sum()

cpdef float premetric(float[:] x1, float[:] x2, float weight=1, sup_dist=None):
    """
    See page 13.
    """
    if sup_dist is None:
        warn("Domain is inferred as 24hr!"
             f"setting sup_dist == 24")
        sup_dist = 24
    return min(1, weight*min(np.fabs(np.subtract(x1,x2)), sup_dist - np.fabs(np.subtract(x1,x2))))

cpdef float empirical_msr(float[:] x, float[:] X, premetric=premetric):
    """
    General (not only local) empirical measure based on a given `premetric`.
    """
    b = np.zeros(X.shape[0])
    for n in range(X.shape[0]):
        b[n] = premetric(x, X[n])
    idcs = np.argsort(b)
    c1 = 1. + 2*np.cumsum(b[idcs])
    c2 = 2*(1.+np.arange(b.shape[0]))*b[idcs]
    M = (c1 > c2).sum()
    mu = (1.+2*b[idcs][:M].sum())/M
    w = np.maximum(-2*b + mu, 0)
    return w
