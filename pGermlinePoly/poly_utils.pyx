from libc.math cimport erf, exp, log, pi, sqrt
import numpy as np


cdef double sqrt2 = sqrt(2.);
cdef double sqrt2pi = sqrt(2*pi);
cdef double logsqrt2pi = log(1/sqrt2pi)

cdef double logsumexp(double[:] x):
    """Cython implementation of the logsumexp trick"""
    cdef int i,n;
    cdef double m = -1e32;
    cdef double c = 0.0;
    n = x.size
    for i in range(n):
        m = max(m,x[i])
    for i in range(n):
        c += exp(x[i] - m)
    return m + log(c)

cdef double log_prior(double [:] l, double[:] a):
    """Cython implementation of the logistic function and log-calculation."""
    cdef int i, n;
    cdef double xk;
    n = l.size
    for i in range(n):
        xk += l[i]*a[i]
    return 1.0 / (1.0 + exp(xk))
