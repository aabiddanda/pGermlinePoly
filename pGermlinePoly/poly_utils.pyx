# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

from libc.math cimport erf, exp, log, log1p, log10, pi, sqrt

import numpy as np
from scipy.optimize import minimize_scalar


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

cdef double beta_logpdf(double x, double a=1.0, double b=1.0):
    """The unscaled log PDF of a specific beta distribution."""
    return (a - 1.0)*log(x) + (b-1.)*log(1.-x)

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
    raw_gl = np.array([0.0, 0.0, 0.0])
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
    cdef double[:] xgl = np.array([0.0, 0.0, 0.0])
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

cpdef double complete_loglik(int K, int J, double[:] lambdas, double[:,:] Theta, double[:,:,:] X, double[:] logll_p):
    """Cython helper for computing the full-data log-likelihood for pGermlinePoly."""
    cdef double logll = 0.0;
    cdef int k,j;
    for k in range(K):
        pi_k = log_prior(lambdas, Theta[k, :])
        log_nonpoly = logll_p[k]
        log_poly = single_var_logll(J, X=X[k,:,:], p=0.5)
        logll += logaddexp(log(pi_k) + log_poly, log(1.0 - pi_k) + log_nonpoly)
    return logll


cpdef double complete_loglik2(int K, int J,  double[:] lambdas, double[:,:] Theta, double[:,:,:] X, int npts=20, double a0=10):
    cdef double logll = 0.0
    cdef double pi0_k;
    cdef int k,j, p;
    cdef double[:] x0;
    cdef double[:] x1;
    cdef double[:] ps = np.linspace(0, 1, npts)
    for k in range(K):
        for p in range(npts):
            x0[p] = beta_logpdf(ps[p], a=a0, b=a0) + single_var_logll(J,X=X[k,:,:], p=ps[p])
            x1[p] = beta_logpdf(ps[p]) + single_var_logll(J,X=X[k,:,:], p=ps[p])
        pi0_k = log_prior(lambdas, Theta[k,:])
        logll += logaddexp(log(pi0_k) + logsumexp(x0), log(1.0 - pi0_k) + logsumexp(x1))
    return logll



cpdef double incomplete_loglik(int K, int J, double[:] lambdas, double[:] gammas_k, double[:,:] Theta, double[:,:,:] X, double[:] logll_p):
    """Cython helper function for computing the incomplete log-likelihood."""
    cdef double logll = 0.0;
    cdef int k, j;
    for k in range(K):
        pi_k = log_prior(lambdas, Theta[k,:])
        log_nonpoly = logll_p[k]
        log_poly = single_var_logll(J, X=X[k,:,:], p=0.5)
        logll += exp(gammas_k[k]) * (log(pi_k) + log_poly) + (1.0 - exp(gammas_k[k])) * (log(1.0 - pi_k) + log_nonpoly)
    return logll
