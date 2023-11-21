from libc.math cimport erf, exp, log, pi, sqrt


cdef double sqrt2 = sqrt(2.);
cdef double sqrt2pi = sqrt(2*pi);
cdef double logsqrt2pi = log(1/sqrt2pi)

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
    cdef double xk;
    n = l.size
    for i in range(n):
        xk += l[i]*a[i]
    return 1.0 / (1.0 + exp(xk))


cpdef double geno_loglik(int alt_reads, int tot_reads, int a1=0, int a2=0, int q=30):
    """Cython implementation of genotype likelihoods under the GATK model.

    Arguments:
        - alt_reads (`int`): number of alternative reads.
        - tot_reads (`int`): number of total reads.
        - a1 (`int`): first allelic state.
        - a2 (`int`): second allelic state.
        - q (`int`): Phred-scaled read quality.
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
