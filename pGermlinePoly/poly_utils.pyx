# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

from libc.math cimport erf, exp, lgamma, log, log1p, log10, pi, sqrt

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

cpdef double logtrapezoid(double[:] x, double[:] ps):
    """Trapezoid rule in log-space."""
    cdef double delta;
    cdef double integrand = 0.0;
    cdef int i, m;
    m = ps.size
    for i in range(1, m):
        delta = log(ps[i] - ps[i-1])
        integrand = logaddexp(integrand, logaddexp(x[i] + delta, x[i-1] + delta))
    integrand = integrand - log(2.0)
    return integrand


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

cpdef double beta_logpdf(double x, double a=1.0, double b=1.0):
    """The unscaled log PDF of a specific beta distribution."""
    if x == 0.0 or x == 1.0:
        return -INFINITY
    else:
        return (a - 1.0)*log(x) + (b - 1.0)*log(1.0 - x) - (lgamma(a) + lgamma(b) - lgamma(a + b))

cdef double[:] phred_rescale(double[:] raw_gl):
    """Perform phred-based rescaling of the genotype likelihoods."""
    cdef double min_gl;
    cdef int i, n;
    cdef double[:] norm_gl = raw_gl;
    min_gl = 1e24
    n = raw_gl.size
    for i in range(n):
        norm_gl[i] = -10*log10(exp(raw_gl[i]))
    for i in range(n):
        min_gl = min(min_gl, norm_gl[i])
    for i in range(n):
        norm_gl[i] = norm_gl[i] - min_gl
    return norm_gl

cdef double geno_gl(int alt_reads, int tot_reads, int a1=0, int a2=0, double q=30.0):
    """Cython implementation of genotype likelihoods under the GATK model.

    Arguments:
        - alt_reads (`int`): number of alternative reads.
        - tot_reads (`int`): number of total reads.
        - a1 (`int`): first allelic state.
        - a2 (`int`): second allelic state.
        - q (`float`): Phred-scaled read quality.
    Returns:
        - gl (`float`): genotype likelihood.
    """
    cdef int i;
    cdef float eps, gl;
    assert a1 in [0,1]
    assert a2 in [0,1]
    assert q > 0
    eps = 10**(-q/10.0)
    gl = 0.0
    for i in range(alt_reads):
        # These are the reads that carry the alternative allele
        gl += log(0.5*((1 - eps)*(a1 == 1) + (eps/3)*(a1 == 0) + (1 - eps)*(a2 == 1) + (eps/3)*(a2 == 0)))
    for i in range(tot_reads - alt_reads):
        # These reads carry only the reference allele
        gl += log(0.5*((1 - eps)*(a1 == 0) + (eps/3)*(a1 == 1)) + 0.5*((1 - eps)*(a2 == 0) + (eps/3)*(a2 == 1)))
    return gl

cpdef double[:] geno_loglik(int alt_reads, int tot_reads, double q=30.0):
    """Actual genotype likelihood computation."""
    cdef double[:] norm_gl;
    cdef double raw_gl[3];
    raw_gl[0] = geno_gl(alt_reads, tot_reads, a1=0, a2=0, q=q)
    raw_gl[1] = 2*geno_gl(alt_reads, tot_reads, a1=1, a2=0, q=q)
    raw_gl[2] = geno_gl(alt_reads, tot_reads, a1=1, a2=1, q=q)
    norm_gl = phred_rescale(raw_gl)
    return norm_gl

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

def mle_est_loglik(K, J, X):
    """
    Estimate the MLE estimate of the allele frequency.

    Store the log-likelihoods under the main allele frequency estimates.
    NOTE: you really only have to do this once and store the outcomes ...
    """
    mle_p = np.zeros(K)
    logll_p = np.zeros(K)
    for k in range(K):
        # This should be on a log-scale here ...
        ll = lambda p: -single_var_logll(J=J, X=X[k,:,:], p=p)
        # NOTE: could we just use the naive MLE estimator here?
        mle_p[k] = minimize_scalar(ll, bounds=(0.0, 1.0)).x
        logll_p[k] = -ll(mle_p[k])
    return mle_p, logll_p

cpdef double posterior_poly(int J, double[:] lambdas, double[:] Theta, double[:,:] X, int npts=20, double a0=5.0):
    """Estimate the posterior probability of germline polymorphism."""
    cdef int k,j,p;
    cdef double denom, num, post_prob;
    cdef double pi0_k;
    cdef double[:] x0 = np.zeros(npts);
    cdef double[:] x1 = np.zeros(npts);
    cdef double[:] ps = np.linspace(0, 1, npts)
    pi0_k = log_prior(lambdas, Theta)
    for p in range(npts):
        x0[p] = beta_logpdf(ps[p], a=a0, b=a0) + single_var_logll(J=J, X=X, p=ps[p])
        x1[p] = beta_logpdf(ps[p], a=1.0, b=1.0) + single_var_logll(J=J, X=X, p=ps[p])
    denom = logaddexp(log(pi0_k) + logsumexp(x0), log(1.0 - pi0_k) + logsumexp(x1))
    num = log(pi0_k) + logsumexp(x0)
    return num - denom


cpdef double complete_loglik(int K, int J, double[:] lambdas, double[:,:] Theta, double[:,:,:] X, int npts=20, double a0=10):
    # NOTE: could we simply evaluate this at the MLE VAF estimate to lower the complexity?
    cdef int k,j,p;
    cdef double logll = 0.0;
    cdef double pi0_k;
    cdef double[:] x0 = np.zeros(npts);
    cdef double[:] x1 = np.zeros(npts);
    cdef double[:] ps = np.linspace(0, 1, npts)
    for k in range(K):
        pi0_k = log_prior(lambdas, Theta[k,:])
        for p in range(npts):
            x0[p] = beta_logpdf(ps[p], a=a0, b=a0) + single_var_logll(J=J, X=X[k,:,:], p=ps[p])
            x1[p] = beta_logpdf(ps[p], a=1.0, b=1.0) + single_var_logll(J=J, X=X[k,:,:], p=ps[p])
        logll += logaddexp(log(pi0_k) + logsumexp(x0), log(1.0 - pi0_k) + logsumexp(x1))
    return logll


cpdef double incomplete_loglik(int K, int J, double[:] lambdas, double[:] gammas_k, double[:,:] Theta, double[:,:,:] X, int npts=20, double a0=10):
    """Cython helper function for computing the incomplete log-likelihood.

    NOTE: I might be using the terms here incorrectly slightly ...
    """
    cdef double logll = 0.0;
    cdef double pi0_k;
    cdef int k, j, p;
    cdef double[:] x0 = np.zeros(npts);
    cdef double[:] x1 = np.zeros(npts);
    cdef double[:] ps = np.linspace(0, 1, npts)
    for k in range(K):
        pi0_k = log_prior(lambdas, Theta[k,:])
        for p in range(npts):
            x0[p] = beta_logpdf(ps[p], a=a0, b=a0) + single_var_logll(J=J, X=X[k,:,:], p=ps[p])
            x1[p] = beta_logpdf(ps[p], a=1.0, b=1.0) + single_var_logll(J=J, X=X[k,:,:], p=ps[p])
        # This should add to the incomplete log-l
        logll += exp(gammas_k[k]) * (log(pi0_k) + logsumexp(x0)) + (1.0 - exp(gammas_k[k])) * (log(1.0 - pi0_k) + logsumexp(x1))
    return logll
