# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

from libc.math cimport exp, expm1, log, log1p, log10, lgamma

cdef extern from "math.h":
    float INFINITY

cdef extern from *:
    """extern double digamma(double);"""
    double digamma(double x) nogil

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
        ll += logaddexp(log(alpha)+logbinomial(ax[i], rx[i], p=0.5),
            log(1 - alpha)+logbinomial(ax[i], rx[i], p=eps))
    return ll

cpdef loglik_ratio(long[:] ax, long[:] rx, double alpha, double eps=1e-3):
    """Evaluate the log-likelihood ratio between these two categories."""
    ll_het = logprob_het(ax, rx)
    ll_somatic = logprob_somatic(ax, rx, alpha=alpha, eps=eps)
    return 2*(ll_somatic - ll_het)

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

cpdef double var_loglik(int alt_reads, int ref_reads, double f, double eps=1e-3):
    """Calculate the likelihood of the underlying reads given VAF."""
    cdef double ref_logll, alt_logll
    ref_logll = ref_reads * log(f*eps + (1 - f)*(1 - eps))
    alt_logll = alt_reads * log(f * (1 - eps) + (1 - f)*eps)
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


# ---------------------------------------------------------------------------
# v2 model primitives (somatic_likelihood_v2.pdf)
# ---------------------------------------------------------------------------

cdef double log_logistic(double x):
    """Numerically stable log(sigma(x))."""
    if x >= 0.0:
        return -log1p(exp(-x))
    else:
        return x - log1p(exp(x))


cdef double log_betabinom(long a, long n, double alpha, double beta):
    """Log BetaBinomial kernel without binomial coefficient.

    Convention matches logbinomial (which also omits the combinatorial term),
    so ratios used in responsibilities are correct.
    """
    cdef long ref = n - a
    return (lgamma(<double>a + alpha) + lgamma(<double>ref + beta)
            - lgamma(<double>n + alpha + beta)
            + lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta))


cdef double logprob_somatic_clone_v2(long a, long n, double logit_phi,
                                      double mu, double kappa):
    """Per-clone somatic log-likelihood P(a_jk, r_jk | z=somatic, phi_j, mu, kappa) (Eq. 1)."""
    cdef double log_phi   = log_logistic(logit_phi)
    cdef double log1m_phi = log_logistic(-logit_phi)
    cdef double log_bin   = logbinomial(a, n - a, 0.5)
    cdef double log_bb    = log_betabinom(a, n, mu * kappa, (1.0 - mu) * kappa)
    return logaddexp(log_phi + log_bin, log1m_phi + log_bb)


cpdef double logprob_somatic_v2(long[:] ax, long[:] rx,
                                 double[:] logit_phi,
                                 double mu=1e-3, double kappa=100.0):
    """Full somatic log-likelihood across all clones (Eq. 2)."""
    cdef int j, J = ax.size
    cdef double ll = 0.0
    for j in range(J):
        ll += logprob_somatic_clone_v2(ax[j], ax[j] + rx[j], logit_phi[j], mu, kappa)
    return ll


cpdef double log_gamma_jk(long a, long n, double logit_phi,
                           double mu, double kappa):
    """Log clone-level responsibility log gamma_jk (Eq. 11)."""
    cdef double log_phi   = log_logistic(logit_phi)
    cdef double log1m_phi = log_logistic(-logit_phi)
    cdef double log_bin   = logbinomial(a, n - a, 0.5)
    cdef double log_bb    = log_betabinom(a, n, mu * kappa, (1.0 - mu) * kappa)
    cdef double log_denom = logaddexp(log_phi + log_bin, log1m_phi + log_bb)
    return log_phi + log_bin - log_denom


cpdef double posterior_poly_v2(long[:] ax, long[:] rx,
                                double[:] logit_phi, double logit_pi,
                                double mu, double kappa):
    """Log posterior P(z_k = het | A_k, R_k) (Eq. 9)."""
    cdef double log_pi      = log_logistic(logit_pi)
    cdef double log1m_pi    = log_logistic(-logit_pi)
    cdef double log_p_het   = logprob_het(ax, rx)
    cdef double log_p_som   = 0.0
    cdef int j, J = ax.size
    for j in range(J):
        log_p_som += logprob_somatic_clone_v2(ax[j], ax[j] + rx[j], logit_phi[j], mu, kappa)
    cdef double log_num   = log_pi + log_p_het
    cdef double log_denom = logaddexp(log_num, log1m_pi + log_p_som)
    return log_num - log_denom


cpdef double observed_loglik_site_v2(long[:] ax, long[:] rx,
                                      double[:] logit_phi, double logit_pi,
                                      double mu, double kappa):
    """Observed data log-likelihood for a single site log P(A_k, R_k)."""
    cdef double log_pi    = log_logistic(logit_pi)
    cdef double log1m_pi  = log_logistic(-logit_pi)
    cdef double log_p_het = logprob_het(ax, rx)
    cdef double log_p_som = 0.0
    cdef int j, J = ax.size
    for j in range(J):
        log_p_som += logprob_somatic_clone_v2(ax[j], ax[j] + rx[j], logit_phi[j], mu, kappa)
    return logaddexp(log_pi + log_p_het, log1m_pi + log_p_som)


cpdef void e_step_all(long[:, :, :] X,
                       double[:, :] logit_phi,
                       double[:] logit_pi,
                       double mu, double kappa,
                       double[:] eta_out,
                       double[:, :] gammas_out):
    """Compute E-step responsibilities for all sites (Eq. 10-11).

    Fills eta_out (M,) with site-level posteriors and gammas_out (M, J)
    with clone-level carrier responsibilities, both in probability space.
    """
    cdef int k, j, M = X.shape[0], J = X.shape[1]
    cdef double log_pi_k, log1m_pi_k, log_p_het_k, log_p_som_k
    cdef double log_eta_num, log_eta_denom
    cdef double log_phi_jk, log1m_phi_jk, log_bin_jk, log_bb_jk

    for k in range(M):
        # log P(A_k, R_k | het)
        log_p_het_k = 0.0
        for j in range(J):
            log_p_het_k += logbinomial(X[k, j, 1], X[k, j, 0], 0.5)

        # log P(A_k, R_k | somatic)
        log_p_som_k = 0.0
        for j in range(J):
            log_p_som_k += logprob_somatic_clone_v2(
                X[k, j, 1], X[k, j, 0] + X[k, j, 1], logit_phi[k, j], mu, kappa
            )

        # Site-level eta_k (Eq. 10)
        log_pi_k   = log_logistic(logit_pi[k])
        log1m_pi_k = log_logistic(-logit_pi[k])
        log_eta_num   = log_pi_k + log_p_het_k
        log_eta_denom = logaddexp(log_eta_num, log1m_pi_k + log_p_som_k)
        eta_out[k]    = exp(log_eta_num - log_eta_denom)

        # Clone-level gamma_jk (Eq. 11)
        for j in range(J):
            log_phi_jk   = log_logistic(logit_phi[k, j])
            log1m_phi_jk = log_logistic(-logit_phi[k, j])
            log_bin_jk   = logbinomial(X[k, j, 1], X[k, j, 0], 0.5)
            log_bb_jk    = log_betabinom(X[k, j, 1], X[k, j, 0] + X[k, j, 1],
                                          mu * kappa, (1.0 - mu) * kappa)
            gammas_out[k, j] = exp(
                log_phi_jk + log_bin_jk
                - logaddexp(log_phi_jk + log_bin_jk, log1m_phi_jk + log_bb_jk)
            )


cpdef double kappa_Q(long[:, :, :] X, double[:, :] gammas,
                      double mu, double kappa):
    """Objective Q(kappa) for the kappa M-step (Eq. 13)."""
    cdef int k, j, M = X.shape[0], J = X.shape[1]
    cdef double Q = 0.0, a, n
    for k in range(M):
        for j in range(J):
            a = X[k, j, 1]
            n = X[k, j, 0] + X[k, j, 1]
            Q += (1.0 - gammas[k, j]) * log_betabinom(
                <long>a, <long>n, mu * kappa, (1.0 - mu) * kappa
            )
    return Q


cpdef double kappa_score(long[:, :, :] X, double[:, :] gammas,
                          double mu, double kappa):
    """Score dQ/dkappa for Brent's method (Eq. 14).

    Uses digamma and gammaln from scipy.special.cython_special.
    """
    cdef int k, j, M = X.shape[0], J = X.shape[1]
    cdef double score = 0.0, a, n, w
    for k in range(M):
        for j in range(J):
            a = X[k, j, 1]
            n = X[k, j, 0] + X[k, j, 1]
            w = 1.0 - gammas[k, j]
            score += w * (
                mu * digamma(a + mu * kappa)
                + (1.0 - mu) * digamma(n - a + (1.0 - mu) * kappa)
                - digamma(n + kappa)
                - mu * digamma(mu * kappa)
                - (1.0 - mu) * digamma((1.0 - mu) * kappa)
                + digamma(kappa)
            )
    return score
