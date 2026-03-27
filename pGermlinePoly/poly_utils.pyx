# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

from libc.math cimport erf, exp, expm1, lgamma, log, log1p, log10, pi, sqrt

import numpy as np
from scipy.optimize import minimize_scalar


cdef extern from "math.h":
    float INFINITY

cpdef double logaddexp(double a, double b):
    """Log-add exponential for two values."""
    cdef double m;
    cdef double c;
    m  = max(a, b)
    c = exp(a - m) + exp(b - m)
    return m + log(c)

cpdef double logsumexp(double[:] x):
    """Cython implementation of the logsumexp trick."""
    cdef int i,n;
    cdef double m = -1e32;
    cdef double c = 0.0;
    n = x.size
    for i in range(n):
        m = max(m,x[i])
    for i in range(n):
        c += exp(x[i] - m)
    return m + log(c)

cdef double log1mexp(double a):
    """Log of 1 - e^-x."""
    if a < 0.693:
        return log(-expm1(-a))
    else:
        return log1p(-exp(-a))

cdef double logbinomial(int alt, int ref, double p):
    """Log-probability mass function of the binomial distribution."""
    return alt * log(p) + ref*log(1. - p)

cpdef double logprob_het(int[:] ax, int[:] rx):
    """Log-probability mass function for a germline heterozygote."""
    cdef int i,j;
    cdef double ll;
    j = ax.size
    ll = 0.0
    for i in range(0, j):
        ll += logbinomial(ax[i], rx[i], p=0.5)
    return ll

cpdef double logprob_somatic(int[:] ax, int[:] rx, double alpha, double eps=1e-3):
    """Log-probability mass function for a somatic mutation."""
    cdef int i, j;
    cdef double ll;
    j = ax.size
    ll = 0
    for i in range(0,j):
        ll += logaddexp(alpha*logbinomial(ax[i],rx[i],p=0.5), (1 - alpha)*logbinomial(ax[i],rx[i],p=eps))
    return ll

cpdef double log_prior(double [:] l, double[:] a):
    """Cython implementation of the logistic function and log-calculation."""
    cdef int i, n;
    cdef double xk = 0.0;
    cdef double prior_p = 0.0;
    n = l.size
    for i in range(0, n):
        xk += l[i]*a[i]
    prior_p = 1.0 / (1.0 + exp(-xk))
    return prior_p

cpdef double var_loglik(int ref_reads, int alt_reads, double f, double q=30.0):
    """Calculate the likelihood of the underlying reads given the allele frequency."""
    cdef double logl = 0.0;
    cdef double ref_logll, alt_logll;
    cdef double eps;
    eps = 10**(-q/10)
    ref_logll = ref_reads * log(f**eps /3 + (1 - f)*(1 - eps))
    alt_logll = alt_reads * log(f * (1 - eps) + (1 - f)**eps / 3 + eps / 3)
    logl = ref_logll + alt_logll
    return logl

def d2_fun(f, x, h=1e-5):
    """Symmetric second derivative function for log-likelihood.

    https://en.wikipedia.org/wiki/Symmetric_derivative#The_second_symmetric_derivative
    """
    return (f(x+h) - 2*f(x) + f(x-h)) / (h**2)

cpdef double single_var_logll(int J, double[:,:] X, double p):
    """Likelihood function for a single-variant."""
    cdef double logll = 0.0;
    cdef int j;
    cdef double xgl[3];
    for j in range(J):
        # Set all of the underlying variables here ...
        xgl[0] = 2*log(1.0-p) + X[j,0]
        xgl[1] = log(2*p*(1-p)) + X[j,1]
        xgl[2] = 2*log(p) + X[j,2]
        logll += logsumexp(xgl)
    return logll

# def mle_est_loglik(K, J, X):
#     """
#     Estimate the MLE estimate of the allele frequency.

#     Store the log-likelihoods under the main allele frequency estimates.
#     NOTE: you really only have to do this once and store the outcomes ...
#     """
#     mle_p = np.zeros(K)
#     logll_p = np.zeros(K)
#     for k in range(K):
#         # This should be on a log-scale here ...
#         ll = lambda p: -single_var_logll(J=J, X=X[k,:,:], p=p)
#         # NOTE: could we just use the naive MLE estimator here?
#         mle_p[k] = minimize_scalar(ll, bounds=(0.0, 1.0)).x
#         logll_p[k] = -ll(mle_p[k])
#     return mle_p, logll_p

# cpdef double posterior_poly(int J, double[:] lambdas, double[:] Theta, double[:,:] X, int npts=20, double a0=5.0):
#     """Estimate the posterior probability of germline polymorphism."""
#     cdef int k,j,p;
#     cdef double denom, num, post_prob;
#     cdef double pi0_k;
#     cdef double[:] x0 = np.zeros(npts);
#     cdef double[:] x1 = np.zeros(npts);
#     cdef double[:] ps = np.linspace(0, 1, npts)
#     pi0_k = log_prior(lambdas, Theta)
#     for p in range(npts):
#         x0[p] = beta_logpdf(ps[p], a=a0, b=a0) + single_var_logll(J=J, X=X, p=ps[p])
#         x1[p] = beta_logpdf(ps[p], a=1.0, b=1.0) + single_var_logll(J=J, X=X, p=ps[p])
#     denom = logaddexp(log(pi0_k) + logsumexp(x0), log(1.0 - pi0_k) + logsumexp(x1))
#     num = log(pi0_k) + logsumexp(x0)
#     return num - denom


# cpdef double complete_loglik(int K, int J, double[:] lambdas, double[:,:] Theta, double[:,:,:] X, int npts=20, double a0=10):
#     # NOTE: could we simply evaluate this at the MLE VAF estimate to lower the complexity?
#     cdef int k,j,p;
#     cdef double logll = 0.0;
#     cdef double pi0_k;
#     cdef double[:] x0 = np.zeros(npts);
#     cdef double[:] x1 = np.zeros(npts);
#     cdef double[:] ps = np.linspace(0, 1, npts)
#     for k in range(K):
#         pi0_k = log_prior(lambdas, Theta[k,:])
#         for p in range(npts):
#             x0[p] = beta_logpdf(ps[p], a=a0, b=a0) + single_var_logll(J=J, X=X[k,:,:], p=ps[p])
#             x1[p] = beta_logpdf(ps[p], a=1.0, b=1.0) + single_var_logll(J=J, X=X[k,:,:], p=ps[p])
#         logll += logaddexp(log(pi0_k) + logsumexp(x0), log(1.0 - pi0_k) + logsumexp(x1))
#     return logll


# cpdef double incomplete_loglik(int K, int J, double[:] lambdas, double[:] gammas_k, double[:,:] Theta, double[:,:,:] X, int npts=20, double a0=10):
#     """Cython helper function for computing the incomplete log-likelihood.
#     """
#     cdef double logll = 0.0;
#     cdef double pi0_k;
#     cdef int k, j, p;
#     cdef double[:] x0 = np.zeros(npts);
#     cdef double[:] x1 = np.zeros(npts);
#     cdef double[:] ps = np.linspace(0, 1, npts)
#     for k in range(K):
#         pi0_k = log_prior(lambdas, Theta[k,:])
#         for p in range(npts):
#             x0[p] = beta_logpdf(ps[p], a=a0, b=a0) + single_var_logll(J=J, X=X[k,:,:], p=ps[p])
#             x1[p] = beta_logpdf(ps[p], a=1.0, b=1.0) + single_var_logll(J=J, X=X[k,:,:], p=ps[p])
#         # This should add to the incomplete log-l
#         logll += exp(gammas_k[k]) * (log(pi0_k) + logsumexp(x0)) + (1.0 - exp(gammas_k[k])) * (log(1.0 - pi0_k) + logsumexp(x1))
#     return logll
