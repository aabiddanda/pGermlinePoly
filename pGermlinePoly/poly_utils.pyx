# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

from libc.math cimport exp, expm1, log, log1p, log10

cdef extern from "math.h":
    float INFINITY

cpdef double logaddexp(double a, double b):
    """Log-add exponential for two values."""
    cdef double m
    cdef double c
    m = max(a, b)
    c = exp(a - m) + exp(b - m)
    return m + log(c)

cpdef double logsumexp(double[:] x):
    """Cython implementation of the logsumexp trick."""
    cdef int i, n
    cdef double m = -1e32
    cdef double c = 0.0
    n = x.size
    for i in range(n):
        m = max(m, x[i])
    for i in range(n):
        c += exp(x[i] - m)
    return m + log(c)

cdef double log1mexp(double a):
    """Log of 1 - e^-x."""
    if a <= 0.693:
        return log(-expm1(-a))
    else:
        return log1p(-exp(-a))

cdef double logbinomial(long alt, long ref, double p):
    """Log-probability mass function of the binomial distribution."""
    return alt * log(p) + ref*log(1. - p)

cpdef double logprob_het(long[:] ax, long[:] rx):
    """Log-probability mass function for a germline heterozygote."""
    cdef int i, j
    cdef double ll
    j = ax.size
    ll = 0.0
    for i in range(0, j):
        ll += logbinomial(ax[i], rx[i], p=0.5)
    return ll

cpdef double logprob_somatic(long[:] ax, long[:] rx, double alpha, double eps=1e-3):
    """Log-probability mass function for a somatic mutation."""
    cdef int i, j
    cdef double ll
    j = ax.size
    ll = 0
    for i in range(0, j):
        ll += logaddexp(alpha*logbinomial(ax[i], rx[i], p=0.5),
            (1 - alpha)*logbinomial(ax[i], rx[i], p=eps))
    return ll

cpdef loglik_ratio(long[:] ax, long[:] rx, double alpha, double eps=1e-3):
    """Evaluate the log-likelihood ratio between these two categories."""
    ll_het = logprob_het(ax, rx)
    ll_somatic = logprob_somatic(ax, rx, alpha=alpha, eps=eps)
    return 2*(ll_het - ll_somatic)

cpdef double log_prior(double[:] l, double[:] a):
    """Cython implementation of the logistic function."""
    cdef int i, n
    cdef double xk = 0.0
    cdef double prior_p = 0.0
    n = l.size
    for i in range(0, n):
        xk += l[i]*a[i]
    prior_p = -log1p(exp(-xk))
    return prior_p


def d2_fun(f, x, h=1e-5):
    """Symmetric second derivative function for log-likelihood.

    https://en.wikipedia.org/wiki/Symmetric_derivative#The_second_symmetric_derivative
    """
    return (f(x+h) - 2*f(x) + f(x-h)) / (h**2)


cpdef double var_loglik(int ref_reads, int alt_reads, double f, double eps=1e-3):
    """Calculate the likelihood of the underlying reads given the allele frequency."""
    cdef double ref_logll, alt_logll
    ref_logll = ref_reads * log(f**eps + (1 - f)*(1 - eps))
    alt_logll = alt_reads * log(f * (1 - eps) + (1 - f)**eps + eps)
    return ref_logll + alt_logll

cpdef double posterior_poly(long[:] ax, long[:] rx,
        double[:] lambdas, double[:] anno,
        double alpha, double eps=1e-3):
    """Calculate the posterior probability of germline polymorphism."""
    cdef double denom, num
    cdef double pi0_k
    pi0_k = log_prior(lambdas, anno)
    p_somatic = logprob_somatic(ax, rx, alpha, eps)
    p_het = logprob_het(ax, rx)
    denom = logaddexp(pi0_k + p_het, log1p(-exp(pi0_k)) + p_somatic)
    num = pi0_k + p_het
    return num - denom

cpdef double complete_loglik(long[:, :, :] X, double[:, :] A, double[:] lambdas, double[:] alpha, double eps=1e-3):
    """Compute the complete data log-likelihood."""
    cdef int m
    cdef double pi0
    cdef double logll = 0.0
    M = X.shape[0]
    for m in range(M):
        pi0 = log_prior(lambdas, A[m, :])
        p_het = logprob_het(X[m, :, 1], X[m, :, 0])
        p_somatic = logprob_somatic(X[m, :, 1], X[m, :, 0], alpha[m], eps)
        logll += logaddexp(pi0 + p_het, log1p(-exp(pi0)) + p_somatic)
    return logll

cpdef double incomplete_loglik(long[:, :, :] X, double[:, :] A, double[:] lambdas, double[:] gammas, double[:] alpha, double eps=1e-3):
    """Compute the incomplete log-likelihood."""
    cdef double logll, pi0
    cdef int m
    M = X.shape[0]
    for k in range(M):
        pi0 = log_prior(lambdas, A[m, :])
        p_het = logprob_het(X[m, :, 1], X[m, :, 0])
        p_somatic = logprob_somatic(X[m, :, 1], X[m, :, 0], alpha[m], eps)
        logll += exp(gammas[m]) * (pi0 + p_het) + (1.0 - exp(gammas[m])) * (log1p(-exp(pi0)) + p_somatic)
    return logll

cdef double[:] phred_rescale(double[:] raw_gl):
    """Perform phred-based rescaling of the genotype likelihoods."""
    cdef double min_gl
    cdef int i, n
    cdef double[:] norm_gl = raw_gl
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
    cdef int i
    cdef float eps, gl
    assert a1 in [0, 1]
    assert a2 in [0, 1]
    assert q > 0
    eps = 10**(-q/10.0)
    gl = 0.0
    for i in range(alt_reads):
        gl += log(0.5*((1 - eps)*(a1 == 1) + (eps/3)*(a1 == 0) +
            (1 - eps)*(a2 == 1) + (eps/3)*(a2 == 0)))
    for i in range(tot_reads - alt_reads):
        gl += log(0.5*((1 - eps)*(a1 == 0) + (eps/3)*(a1 == 1)) +
            0.5*((1 - eps)*(a2 == 0) + (eps/3)*(a2 == 1)))
    return gl

cpdef double[:] geno_loglik(int alt_reads, int tot_reads, double q=30.0):
    """Actual genotype likelihood computation."""
    cdef double[:] norm_gl
    cdef double raw_gl[3]
    raw_gl[0] = geno_gl(alt_reads, tot_reads, a1=0, a2=0, q=q)
    raw_gl[1] = 2*geno_gl(alt_reads, tot_reads, a1=1, a2=0, q=q)
    raw_gl[2] = geno_gl(alt_reads, tot_reads, a1=1, a2=1, q=q)
    norm_gl = phred_rescale(raw_gl)
    return norm_gl
