from libc.math cimport erf, exp, log, log10, pi, sqrt

import numpy as np


cdef double sqrt2 = sqrt(2.);
cdef double sqrt2pi = sqrt(2*pi);
cdef double logsqrt2pi = log(1/sqrt2pi)


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

# Should we add in an intercept term here?
cpdef double log_prior(double [:] l, double[:] a, double eps=1e-9):
    """Cython implementation of the logistic function and log-calculation."""
    cdef int i, n;
    cdef double xk, prior_p;
    n = l.size
    prior_p = 0.0
    for i in range(0, n):
        xk += l[i]*a[i]
    # avoids some potential errors in here with under flow
    prior_p = max(1.0 / (1.0 + exp(-xk)), eps)
    prior_p = min(prior_p, 1.0 - eps)
    return prior_p


cdef double[:] phred_rescale(double[:] raw_gl):
    """Perform phred-based rescaling of the genotype likelihoods."""
    cdef double min_gl;
    cdef int i, n;
    min_gl = 1e24
    n = raw_gl.size
    norm_gl = [-log10(exp(x)) for x in raw_gl]
    for i in range(n):
        min_gl = min(min_gl, norm_gl[i])
    norm_gl = np.array([x - min_gl for x in norm_gl])
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

cpdef double complete_loglik(int K, int J, double[:] lambdas, double[:,:] Theta, double[:,:,:] X):
    """Cython helper for computing the full-data log-likelihood for pGermlinePoly."""
    cdef double logll = 0.0;
    cdef int k,j;

    for k in range(K):
        pi_k = log_prior(lambdas, Theta[k, :])
        for j in range(J):
            # Compute the likelihood as a sum across sites
            logll += logaddexp(log(pi_k) + logsumexp(X[k, j, 1:-1]), log(1.0 - pi_k) + logaddexp(X[k, j, 0], X[k, j, -1]))
    return logll

cpdef double incomplete_loglik(int K, int J, double[:] lambdas, double[:] gammas_k, double[:,:] Theta, double[:,:,:] X):
    """Cython helper function for computing the incomplete log-likelihood."""
    cdef double logll = 0.0;
    cdef int k, j;
    for k in range(K):
        pi_k = log_prior(lambdas, Theta[k,:])
        logll_poly = 0.0
        logll_nonpoly = 0.0
        for j in range(J):
            logll += exp(gammas_k[k]) * (log(pi_k) + logsumexp(X[k, j, 1:-1])) + (1.0 - exp(gammas_k[k])) * (log(1.0 - pi_k) + logaddexp(X[k,j,0], X[k,j,-1]))
    return logll
