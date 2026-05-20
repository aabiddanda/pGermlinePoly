# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

from libc.math cimport exp, expm1, log, log1p, log10, lgamma
from cython.parallel cimport prange

cdef extern from "math.h":
    float INFINITY

cdef double digamma(double x) nogil:
    """Compute the digamma function using upward recurrence and asymptotic expansion.

    Shifts x up via the recurrence psi(x) = psi(x+1) - 1/x until x >= 6,
    then applies the Stirling asymptotic series.

    Parameters
    ----------
    x : double
        Argument of the digamma function. Must be positive.

    Returns
    -------
    double
        Value of psi(x).

    Notes
    -----
    Asymptotic series used for x >= 6::

        psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4)
                  - 1/(252x^6) + 1/(240x^8) - 1/(132x^10)
    """
    cdef double result = 0.0
    cdef double y, y2
    # Shift x up: psi(x) = psi(x+1) - 1/x
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic (Stirling) series: psi(x) ~ ln(x) - 1/(2x) - B_{2n}/(2n*x^{2n})
    # Coefficients: -1/12x^2 + 1/120x^4 - 1/252x^6 + 1/240x^8 - 1/132x^10
    y = 1.0 / x
    y2 = y * y
    result += (log(x) - 0.5 * y
               - y2 * (1.0/12.0
               - y2 * (1.0/120.0
               - y2 * (1.0/252.0
               - y2 * (1.0/240.0
               - y2 / 132.0)))))
    return result

cpdef double logaddexp(double a, double b) nogil:
    """Compute log(exp(a) + exp(b)) in a numerically stable way.

    Parameters
    ----------
    a : double
        First log-space value.
    b : double
        Second log-space value.

    Returns
    -------
    double
        log(exp(a) + exp(b)).
    """
    cdef double m
    cdef double c
    m = max(a, b)
    c = exp(a - m) + exp(b - m)
    return m + log(c)

cpdef double logsumexp(double[:] x):
    """Compute log(sum(exp(x))) using the logsumexp trick for numerical stability.

    Parameters
    ----------
    x : double[:]
        1-D array of log-space values, shape (N,).

    Returns
    -------
    double
        log(sum_i exp(x[i])).
    """
    cdef int i, n
    cdef double m = -1e32
    cdef double c = 0.0
    n = x.size
    for i in range(n):
        m = max(m, x[i])
    for i in range(n):
        c += exp(x[i] - m)
    return m + log(c)

cdef double log1mexp(double a) nogil:
    """Compute log(1 - exp(-a)) in a numerically stable way.

    Parameters
    ----------
    a : double
        Non-negative value.

    Returns
    -------
    double
        log(1 - exp(-a)).

    Notes
    -----
    Uses ``log(-expm1(-a))`` when ``a <= log(2)`` and ``log1p(-exp(-a))``
    otherwise, following Machler (2012) to avoid cancellation near zero.
    """
    if a <= 0.693:
        return log(-expm1(-a))
    else:
        return log1p(-exp(-a))

cdef double logbinomial(long alt, long ref, double p) nogil:
    """Compute the log binomial probability kernel (no combinatorial term).

    Returns ``alt * log(p) + ref * log(1 - p)``. The binomial coefficient
    is omitted, matching the convention in ``log_betabinom`` so that
    responsibility ratios are correct.

    Parameters
    ----------
    alt : long
        Number of successes (alternative allele reads).
    ref : long
        Number of failures (reference allele reads).
    p : double
        Success probability.

    Returns
    -------
    double
        alt * log(p) + ref * log(1 - p).
    """
    return alt * log(p) + ref*log(1. - p)

cpdef double logprob_het(long[:] ax, long[:] rx):
    """Compute the log-likelihood of reads under a germline heterozygote model.

    Evaluates the binomial log-likelihood at p = 0.5 for each clone and sums
    over all J clones.

    Parameters
    ----------
    ax : long[:]
        Alternative read counts per clone, shape (J,).
    rx : long[:]
        Reference read counts per clone, shape (J,).

    Returns
    -------
    double
        sum_j logBinom(ax[j], rx[j]; p=0.5).
    """
    cdef int i, j
    cdef double ll
    j = ax.size
    ll = 0.0
    for i in range(0, j):
        ll += logbinomial(ax[i], rx[i], p=0.5)
    return ll

cpdef double logprob_somatic(long[:] ax, long[:] rx, double alpha, double eps=1e-3):
    """Compute the log-likelihood of reads under a somatic mutation model.

    Each clone is modelled as a two-component mixture: with probability
    ``alpha`` the clone carries the mutation (binomial at p=0.5), otherwise
    it shows only sequencing error (binomial at p=``eps``). Log-likelihoods
    are summed over all J clones.

    Parameters
    ----------
    ax : long[:]
        Alternative read counts per clone, shape (J,).
    rx : long[:]
        Reference read counts per clone, shape (J,).
    alpha : double
        Mixture weight for the carrier component.
    eps : double, optional
        Sequencing error rate. Default is 1e-3.

    Returns
    -------
    double
        sum_j log[alpha * Binom(0.5) + (1-alpha) * Binom(eps)].
    """
    cdef int i, j
    cdef double ll
    j = ax.size
    ll = 0
    for i in range(0, j):
        ll += logaddexp(log(alpha)+logbinomial(ax[i], rx[i], p=0.5),
            log(1 - alpha)+logbinomial(ax[i], rx[i], p=eps))
    return ll

cpdef loglik_ratio(long[:] ax, long[:] rx, double alpha, double eps=1e-3):
    """Compute the log-likelihood ratio statistic 2*(LL_somatic - LL_het).

    Parameters
    ----------
    ax : long[:]
        Alternative read counts per clone, shape (J,).
    rx : long[:]
        Reference read counts per clone, shape (J,).
    alpha : double
        Mixture weight for the somatic carrier component.
    eps : double, optional
        Sequencing error rate. Default is 1e-3.

    Returns
    -------
    double
        2 * (logprob_somatic - logprob_het).
    """
    ll_het = logprob_het(ax, rx)
    ll_somatic = logprob_somatic(ax, rx, alpha=alpha, eps=eps)
    return 2*(ll_somatic - ll_het)

cpdef double log_prior(double[:] l, double[:] a):
    """Compute the log prior probability of germline status under a logistic model.

    Evaluates ``log sigma(l . a) = -log(1 + exp(-l . a))``, where sigma is
    the logistic (sigmoid) function and ``l . a`` is the dot product of the
    weight vector and site annotation vector.

    Parameters
    ----------
    l : double[:]
        Logistic regression weight vector (lambda), shape (P,).
    a : double[:]
        Site annotation feature vector, shape (P,).

    Returns
    -------
    double
        log sigma(l . a).
    """
    cdef int i, n
    cdef double xk = 0.0
    cdef double prior_p = 0.0
    n = l.size
    for i in range(0, n):
        xk += l[i]*a[i]
    prior_p = -log1p(exp(-xk))
    return prior_p

cpdef double var_loglik(int alt_reads, int ref_reads, double f, double eps=1e-3):
    """Compute the log-likelihood of observed reads given a variant allele frequency.

    Models reads as a mixture of true variant signal and sequencing error,
    parameterised by VAF ``f`` and error rate ``eps``.

    Parameters
    ----------
    alt_reads : int
        Number of alternative allele reads.
    ref_reads : int
        Number of reference allele reads.
    f : double
        Variant allele frequency (VAF), in [0, 1].
    eps : double, optional
        Sequencing error rate. Default is 1e-3.

    Returns
    -------
    double
        log L(f) = ref_reads * log(f*eps + (1-f)*(1-eps))
                   + alt_reads * log(f*(1-eps) + (1-f)*eps).
    """
    cdef double ref_logll, alt_logll
    ref_logll = ref_reads * log(f*eps + (1 - f)*(1 - eps))
    alt_logll = alt_reads * log(f * (1 - eps) + (1 - f)*eps)
    return ref_logll + alt_logll

cpdef double posterior_poly(long[:] ax, long[:] rx,
        double[:] lambdas, double[:] anno,
        double alpha, double eps=1e-3):
    """Compute the log posterior probability of germline polymorphism.

    Combines the logistic prior with the heterozygote and somatic likelihoods
    using Bayes' rule and returns ``log P(z = het | A, R)``.

    Parameters
    ----------
    ax : long[:]
        Alternative read counts per clone, shape (J,).
    rx : long[:]
        Reference read counts per clone, shape (J,).
    lambdas : double[:]
        Logistic regression weight vector, shape (P,).
    anno : double[:]
        Site annotation feature vector, shape (P,).
    alpha : double
        Mixture weight for the somatic carrier component.
    eps : double, optional
        Sequencing error rate. Default is 1e-3.

    Returns
    -------
    double
        log P(z = het | A, R).
    """
    cdef double denom, num
    cdef double pi0_k
    pi0_k = log_prior(lambdas, anno)
    p_somatic = logprob_somatic(ax, rx, alpha, eps)
    p_het = logprob_het(ax, rx)
    denom = logaddexp(pi0_k + p_het, log1p(-exp(pi0_k)) + p_somatic)
    num = pi0_k + p_het
    return num - denom

cpdef double complete_loglik(long[:, :, :] X, double[:, :] A, double[:] lambdas, double[:] alpha, double eps=1e-3):
    """Compute the complete-data log-likelihood summed over all M sites.

    Parameters
    ----------
    X : long[:, :, :]
        Read count array of shape (M, J, 2); X[m, j, 0] = ref reads,
        X[m, j, 1] = alt reads.
    A : double[:, :]
        Site annotation matrix, shape (M, P).
    lambdas : double[:]
        Logistic regression weight vector, shape (P,).
    alpha : double[:]
        Per-site somatic carrier mixture weights, shape (M,).
    eps : double, optional
        Sequencing error rate. Default is 1e-3.

    Returns
    -------
    double
        sum_m log P(A_m, R_m | lambdas, alpha_m, eps).
    """
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
    """Compute the expected complete-data log-likelihood (Q function) for the EM algorithm.

    Weights each site's contribution by the current E-step responsibilities
    ``gammas[k]``, forming the Q function used in the EM M-step.

    Parameters
    ----------
    X : long[:, :, :]
        Read count array of shape (M, J, 2); X[k, j, 0] = ref reads,
        X[k, j, 1] = alt reads.
    A : double[:, :]
        Site annotation matrix, shape (M, P).
    lambdas : double[:]
        Logistic regression weight vector, shape (P,).
    gammas : double[:]
        Log-space site-level responsibilities from the E-step, shape (M,).
    alpha : double[:]
        Per-site somatic carrier mixture weights, shape (M,).
    eps : double, optional
        Sequencing error rate. Default is 1e-3.

    Returns
    -------
    double
        Q function value summed over all M sites.
    """
    cdef double logll = 0.0, pi0
    cdef int k, M = X.shape[0]
    for k in range(M):
        pi0 = log_prior(lambdas, A[k, :])
        p_het = logprob_het(X[k, :, 1], X[k, :, 0])
        p_somatic = logprob_somatic(X[k, :, 1], X[k, :, 0], alpha[k], eps)
        logll += exp(gammas[k]) * (pi0 + p_het) + (1.0 - exp(gammas[k])) * (log1p(-exp(pi0)) + p_somatic)
    return logll

cdef double[:] phred_rescale(double[:] raw_gl):
    """Rescale log-space genotype likelihoods to Phred scale and normalize.

    Converts each entry to -10 * log10(prob), then subtracts the minimum
    value so that the best genotype has a Phred score of 0.

    Parameters
    ----------
    raw_gl : double[:]
        Log-probability genotype likelihoods, shape (3,). Modified in place.

    Returns
    -------
    double[:]
        Phred-scaled, minimum-normalized genotype likelihoods, shape (3,).
    """
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
    """Compute a single diploid genotype log-likelihood under the GATK error model.

    Parameters
    ----------
    alt_reads : int
        Number of alternative allele reads.
    tot_reads : int
        Total number of reads (alt + ref).
    a1 : int, optional
        First allelic state (0 = ref, 1 = alt). Default is 0.
    a2 : int, optional
        Second allelic state (0 = ref, 1 = alt). Default is 0.
    q : double, optional
        Phred-scaled per-base read quality. Must be positive. Default is 30.0.

    Returns
    -------
    double
        log P(reads | genotype (a1, a2), quality q).
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
    """Compute Phred-scaled genotype likelihoods for the three diploid genotypes.

    Evaluates the GATK genotype likelihood model for homozygous reference
    (0/0), heterozygous (0/1), and homozygous alternative (1/1), then
    converts to Phred scale and normalizes so the best genotype scores 0.

    Parameters
    ----------
    alt_reads : int
        Number of alternative allele reads.
    tot_reads : int
        Total number of reads (alt + ref).
    q : double, optional
        Phred-scaled per-base read quality. Default is 30.0.

    Returns
    -------
    double[:]
        Phred-scaled PL values for genotypes [0/0, 0/1, 1/1], shape (3,),
        with the minimum PL value set to 0.
    """
    cdef double[:] norm_gl
    cdef double raw_gl[3]
    raw_gl[0] = geno_gl(alt_reads, tot_reads, a1=0, a2=0, q=q)
    raw_gl[1] = 2*geno_gl(alt_reads, tot_reads, a1=1, a2=0, q=q)
    raw_gl[2] = geno_gl(alt_reads, tot_reads, a1=1, a2=1, q=q)
    norm_gl = phred_rescale(raw_gl)
    return norm_gl


cpdef void geno_loglik_2d(
    long[:, :] alt_reads,
    long[:, :] tot_reads,
    double[:, :, :] out,
    double q=30.0,
):
    """Fill *out* with Phred-scaled PLs for a 2-D array of (alt, tot) pairs.

    Equivalent to calling ``geno_loglik(alt_reads[i,j], tot_reads[i,j], q)``
    for each (i, j) and storing the result in ``out[i, j, :]``. Precomputes
    per-base log probabilities once and loops in C over the (M, J) grid.

    Parameters
    ----------
    alt_reads : long[:, :]
        Alt-read counts, shape (M, J).
    tot_reads : long[:, :]
        Total read counts, shape (M, J).
    out : double[:, :, :]
        Pre-allocated output buffer, shape (M, J, 3). Modified in place.
    q : double, optional
        Phred-scaled read quality. Default 30.
    """
    cdef int M = alt_reads.shape[0]
    cdef int J = alt_reads.shape[1]
    cdef int i, j
    cdef double eps, lp_err, lp_het, lp_ref, gl00, gl01, gl11, gl_min
    eps = 10.0 ** (-q / 10.0)
    lp_err = log(eps / 3.0)
    lp_het = log(0.5 * (1.0 - eps) + eps / 6.0)
    lp_ref = log(1.0 - eps)
    for i in range(M):
        for j in range(J):
            gl00 = alt_reads[i, j] * lp_err + (tot_reads[i, j] - alt_reads[i, j]) * lp_ref
            gl01 = 2.0 * tot_reads[i, j] * lp_het
            gl11 = alt_reads[i, j] * lp_ref + (tot_reads[i, j] - alt_reads[i, j]) * lp_err
            out[i, j, 0] = -10.0 * log10(exp(gl00))
            out[i, j, 1] = -10.0 * log10(exp(gl01))
            out[i, j, 2] = -10.0 * log10(exp(gl11))
            gl_min = out[i, j, 0]
            if out[i, j, 1] < gl_min:
                gl_min = out[i, j, 1]
            if out[i, j, 2] < gl_min:
                gl_min = out[i, j, 2]
            out[i, j, 0] -= gl_min
            out[i, j, 1] -= gl_min
            out[i, j, 2] -= gl_min


# ---------------------------------------------------------------------------
# Beta-Binomial error model primitives
# ---------------------------------------------------------------------------

cdef double log_logistic(double x) nogil:
    """Compute the log-sigmoid log(sigma(x)) in a numerically stable way.

    Parameters
    ----------
    x : double
        Input value (logit).

    Returns
    -------
    double
        log(1 / (1 + exp(-x))).

    Notes
    -----
    Uses ``-log1p(exp(-x))`` for x >= 0 and ``x - log1p(exp(x))`` for x < 0
    to avoid overflow for large |x|.
    """
    if x >= 0.0:
        return -log1p(exp(-x))
    else:
        return x - log1p(exp(x))


cdef double log_betabinom(long a, long n, double alpha, double beta) nogil:
    """Compute the log Beta-Binomial probability kernel (binomial coefficient omitted).

    The binomial coefficient is dropped to match the convention in
    ``logbinomial``, so responsibility ratios computed from the two kernels
    are numerically correct.

    Parameters
    ----------
    a : long
        Alternative allele read count.
    n : long
        Total read depth (alt + ref).
    alpha : double
        First Beta shape parameter (typically ``mu * kappa``).
    beta : double
        Second Beta shape parameter (typically ``(1 - mu) * kappa``).

    Returns
    -------
    double
        lgamma(a + alpha) + lgamma(ref + beta) - lgamma(n + alpha + beta)
        + lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta),
        where ref = n - a.
    """
    cdef long ref = n - a
    return (lgamma(<double>a + alpha) + lgamma(<double>ref + beta)
            - lgamma(<double>n + alpha + beta)
            + lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta))


cdef double log_betabinom_prenorm(
        long a, long n, double alpha, double beta,
        double log_norm) nogil:
    """log_betabinom with a precomputed normalization constant.

    Identical to ``log_betabinom`` but skips recomputing the three lgamma
    terms that depend only on alpha and beta (not on the data).  Call this
    inside clone-level loops where alpha and beta are fixed across clones.

    Parameters
    ----------
    a : long
        Alternative allele read count.
    n : long
        Total read depth (alt + ref).
    alpha : double
        First Beta shape parameter.
    beta : double
        Second Beta shape parameter.
    log_norm : double
        Precomputed lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta).

    Returns
    -------
    double
        lgamma(a + alpha) + lgamma(ref + beta) - lgamma(n + alpha + beta) + log_norm.
    """
    cdef long ref = n - a
    return (lgamma(<double>a + alpha) + lgamma(<double>ref + beta)
            - lgamma(<double>n + alpha + beta) + log_norm)


cpdef double sum_log_betabinom(
        long[:] alt_reads, long[:] ref_reads,
        double alpha, double beta):
    """Sum the log Beta-Binomial kernel over clones for a single site.

    Computes sum_j log_betabinom(alt_j, alt_j + ref_j, alpha, beta).
    The binomial coefficient is omitted — it is constant with respect to
    the rho parameter being optimised and therefore does not affect the argmax.

    Parameters
    ----------
    alt_reads : long[:]
        Alternative allele read counts per clone, shape (J,).
    ref_reads : long[:]
        Reference allele read counts per clone, shape (J,).
    alpha : double
        First Beta shape parameter (e.g. phat * (1 - rho) / rho).
    beta : double
        Second Beta shape parameter (e.g. (1 - phat) * (1 - rho) / rho).

    Returns
    -------
    double
        sum_j log_betabinom(alt_j, n_j, alpha, beta).
    """
    cdef int j, J = alt_reads.shape[0]
    cdef double acc = 0.0
    cdef double log_norm = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
    for j in range(J):
        acc += log_betabinom_prenorm(alt_reads[j], alt_reads[j] + ref_reads[j],
                                     alpha, beta, log_norm)
    return acc


cdef double logprob_somatic_clone(
        long a, long n, double logit_phi,
        double mu, double kappa) nogil:
    """Compute the per-clone somatic log-likelihood under the Beta-Binomial error model.

    Evaluates log P(a_jk | z=somatic, phi_j, mu, kappa) as a two-component
    mixture: a binomial carrier term and a Beta-Binomial error term, as in
    Eq. 1 of the model.

    Parameters
    ----------
    a : long
        Alternative allele read count for this clone at this site.
    n : long
        Total read depth for this clone (alt + ref).
    logit_phi : double
        Logit-transformed clone carrier probability logit(phi_j).
    mu : double
        Mean of the Beta-Binomial error distribution.
    kappa : double
        Concentration parameter of the Beta-Binomial distribution.

    Returns
    -------
    double
        log[phi_j * Binom(a; n, 0.5) + (1 - phi_j) * BetaBinom(a; n, mu*kappa, (1-mu)*kappa)].
    """
    cdef double log_phi = log_logistic(logit_phi)
    cdef double log1m_phi = log_logistic(-logit_phi)
    cdef double log_bin = logbinomial(a, n - a, 0.5)
    cdef double log_bb = log_betabinom(a, n, mu * kappa, (1.0 - mu) * kappa)
    return logaddexp(log_phi + log_bin, log1m_phi + log_bb)


cpdef double logprob_somatic_bb(
        long[:] ax, long[:] rx,
        double[:] logit_phi,
        double mu=1e-3, double kappa=100.0):
    """Compute the full somatic log-likelihood summed across all clones.

    Sums per-clone somatic log-likelihoods over J clones for a single site,
    as in Eq. 2 of the model.

    Parameters
    ----------
    ax : long[:]
        Alternative read counts per clone, shape (J,).
    rx : long[:]
        Reference read counts per clone, shape (J,).
    logit_phi : double[:]
        Logit-transformed clone carrier probabilities, shape (J,).
    mu : double, optional
        Mean of the Beta-Binomial error distribution. Default is 1e-3.
    kappa : double, optional
        Concentration parameter of the Beta-Binomial distribution. Default is 100.0.

    Returns
    -------
    double
        sum_j logprob_somatic_clone(ax[j], ax[j]+rx[j], logit_phi[j], mu, kappa).
    """
    cdef int j, J = ax.size
    cdef double ll = 0.0
    for j in range(J):
        ll += logprob_somatic_clone(ax[j], ax[j] + rx[j], logit_phi[j], mu, kappa)
    return ll


cpdef double log_gamma_jk(
        long a, long n, double logit_phi,
        double mu, double kappa):
    """Compute the log clone-level carrier responsibility log gamma_jk.

    Returns the log posterior probability that clone j carries the somatic
    mutation at site k, i.e. log P(carrier | a_jk), as in Eq. 11.

    Parameters
    ----------
    a : long
        Alternative allele read count for clone j at site k.
    n : long
        Total read depth for clone j at site k (alt + ref).
    logit_phi : double
        Logit-transformed carrier probability for clone j, logit(phi_j).
    mu : double
        Mean of the Beta-Binomial error distribution.
    kappa : double
        Concentration parameter of the Beta-Binomial distribution.

    Returns
    -------
    double
        log gamma_jk = log phi_j + log Binom(a; n, 0.5)
                       - logaddexp(log phi_j + log Binom, log(1-phi_j) + log BetaBinom).
    """
    cdef double log_phi = log_logistic(logit_phi)
    cdef double log1m_phi = log_logistic(-logit_phi)
    cdef double log_bin = logbinomial(a, n - a, 0.5)
    cdef double log_bb = log_betabinom(a, n, mu * kappa, (1.0 - mu) * kappa)
    cdef double log_denom = logaddexp(log_phi + log_bin, log1m_phi + log_bb)
    return log_phi + log_bin - log_denom


cpdef double log_posterior_germline(long[:] ax, long[:] rx,
                                double[:] logit_phi, double logit_pi,
                                double mu, double kappa):
    """Compute the log posterior probability that site k is a germline heterozygote.

    Returns log P(z_k = het | A_k, R_k) by combining the site-level prior
    pi_k with the heterozygote and somatic likelihoods summed over J clones,
    as in Eq. 9.

    Parameters
    ----------
    ax : long[:]
        Alternative read counts per clone, shape (J,).
    rx : long[:]
        Reference read counts per clone, shape (J,).
    logit_phi : double[:]
        Logit-transformed clone carrier probabilities, shape (J,).
    logit_pi : double
        Logit-transformed site-level germline prior probability logit(pi_k).
    mu : double
        Mean of the Beta-Binomial error distribution.
    kappa : double
        Concentration parameter of the Beta-Binomial distribution.

    Returns
    -------
    double
        log P(z_k = het | A_k, R_k).
    """
    cdef double log_pi = log_logistic(logit_pi)
    cdef double log1m_pi = log_logistic(-logit_pi)
    cdef double log_p_het = logprob_het(ax, rx)
    cdef double log_p_som = 0.0
    cdef int j, J = ax.size
    for j in range(J):
        log_p_som += logprob_somatic_clone(ax[j], ax[j] + rx[j], logit_phi[j], mu, kappa)
    cdef double log_num = log_pi + log_p_het
    cdef double log_denom = logaddexp(log_num, log1m_pi + log_p_som)
    return log_num - log_denom


cpdef double observed_loglik_site(
        long[:] ax, long[:] rx,
        double[:] logit_phi, double logit_pi,
        double mu, double kappa):
    """Compute the observed-data log-likelihood for a single site.

    Marginalizes over the latent class z_k to give
    log P(A_k, R_k) = logaddexp(log pi_k + log P_het, log(1-pi_k) + log P_som).

    Parameters
    ----------
    ax : long[:]
        Alternative read counts per clone, shape (J,).
    rx : long[:]
        Reference read counts per clone, shape (J,).
    logit_phi : double[:]
        Logit-transformed clone carrier probabilities, shape (J,).
    logit_pi : double
        Logit-transformed site-level germline prior probability logit(pi_k).
    mu : double
        Mean of the Beta-Binomial error distribution.
    kappa : double
        Concentration parameter of the Beta-Binomial distribution.

    Returns
    -------
    double
        log P(A_k, R_k).
    """
    cdef double log_pi = log_logistic(logit_pi)
    cdef double log1m_pi = log_logistic(-logit_pi)
    cdef double log_p_het = logprob_het(ax, rx)
    cdef double log_p_som = 0.0
    cdef int j, J = ax.size
    for j in range(J):
        log_p_som += logprob_somatic_clone(ax[j], ax[j] + rx[j], logit_phi[j], mu, kappa)
    return logaddexp(log_pi + log_p_het, log1m_pi + log_p_som)


cdef double _log_p_het_row(long[:, :, :] X, int k, int J) nogil:
    """Sum the heterozygote log-likelihood over all clones for site k.

    Parameters
    ----------
    X : long[:, :, :]
        Read count array of shape (M, J, 2).
    k : int
        Site index.
    J : int
        Number of clones.

    Returns
    -------
    double
        sum_j logBinom(X[k,j,1], X[k,j,0]; p=0.5).
    """
    cdef int j
    cdef double acc = 0.0
    for j in range(J):
        acc += logbinomial(X[k, j, 1], X[k, j, 0], 0.5)
    return acc


cdef double _log_p_som_row(
        long[:, :, :] X, double[:, :] logit_phi,
        int k, int J,
        double alpha, double beta, double log_norm) nogil:
    """Sum the somatic log-likelihood over all clones for site k.

    Parameters
    ----------
    X : long[:, :, :]
        Read count array of shape (M, J, 2).
    logit_phi : double[:, :]
        Logit-transformed clone carrier probabilities, shape (M, J).
    k : int
        Site index.
    J : int
        Number of clones.
    alpha : double
        mu * kappa (first Beta shape parameter).
    beta : double
        (1 - mu) * kappa (second Beta shape parameter).
    log_norm : double
        Precomputed lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta).

    Returns
    -------
    double
        sum_j logprob_somatic_clone for site k.
    """
    cdef int j
    cdef double acc = 0.0, log_phi, log1m_phi, log_bin, log_bb
    for j in range(J):
        log_phi = log_logistic(logit_phi[k, j])
        log1m_phi = log_logistic(-logit_phi[k, j])
        log_bin = logbinomial(X[k, j, 1], X[k, j, 0], 0.5)
        log_bb = log_betabinom_prenorm(
            X[k, j, 1], X[k, j, 0] + X[k, j, 1],
            alpha, beta, log_norm)
        acc += logaddexp(log_phi + log_bin, log1m_phi + log_bb)
    return acc


cdef void _fill_gammas_row(
        long[:, :, :] X, double[:, :] logit_phi,
        double[:, :] gammas_out,
        int k, int J,
        double alpha, double beta, double log_norm) noexcept nogil:
    """Fill gammas_out[k, :] with clone-level carrier responsibilities for site k.

    Writes exp(log_gamma_jk) into gammas_out[k, j] for each clone j.

    Parameters
    ----------
    X : long[:, :, :]
        Read count array of shape (M, J, 2).
    logit_phi : double[:, :]
        Logit-transformed clone carrier probabilities, shape (M, J).
    gammas_out : double[:, :]
        Output responsibility array, shape (M, J). Modified in place.
    k : int
        Site index.
    J : int
        Number of clones.
    alpha : double
        mu * kappa (first Beta shape parameter).
    beta : double
        (1 - mu) * kappa (second Beta shape parameter).
    log_norm : double
        Precomputed lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta).
    """
    cdef int j
    cdef long n_kj
    cdef double log_phi_jk, log1m_phi_jk, log_bin_jk, log_bb_jk
    for j in range(J):
        n_kj = X[k, j, 0] + X[k, j, 1]
        log_phi_jk = log_logistic(logit_phi[k, j])
        log1m_phi_jk = log_logistic(-logit_phi[k, j])
        log_bin_jk = logbinomial(X[k, j, 1], X[k, j, 0], 0.5)
        log_bb_jk = log_betabinom_prenorm(X[k, j, 1], n_kj, alpha, beta, log_norm)
        gammas_out[k, j] = exp(
            log_phi_jk + log_bin_jk
            - logaddexp(log_phi_jk + log_bin_jk, log1m_phi_jk + log_bb_jk)
        )


cpdef void e_step_all(
        long[:, :, :] X,
        double[:, :] logit_phi,
        double[:] logit_pi,
        double mu, double kappa,
        double[:] eta_out,
        double[:, :] gammas_out):
    """Compute E-step responsibilities for all sites in parallel.

    Fills ``eta_out`` with site-level posterior probabilities of germline
    heterozygosity and ``gammas_out`` with clone-level carrier
    responsibilities, both in probability space. Sites are processed in
    parallel via OpenMP ``prange``, as described in Eqs. 10-11.

    Parameters
    ----------
    X : long[:, :, :]
        Read count array of shape (M, J, 2); X[k, j, 0] = ref reads,
        X[k, j, 1] = alt reads.
    logit_phi : double[:, :]
        Logit-transformed clone carrier probabilities, shape (M, J).
    logit_pi : double[:]
        Logit-transformed site-level germline prior probabilities, shape (M,).
    mu : double
        Mean of the Beta-Binomial error distribution.
    kappa : double
        Concentration parameter of the Beta-Binomial distribution.
    eta_out : double[:]
        Output array for P(z_k = het | data), shape (M,). Modified in place.
    gammas_out : double[:, :]
        Output array for clone-level carrier responsibilities, shape (M, J).
        Modified in place.
    """
    cdef int k, M = X.shape[0], J = X.shape[1]
    cdef double log_pi_k, log1m_pi_k, log_p_het_k, log_p_som_k
    cdef double log_eta_num, log_eta_denom
    cdef double alpha = mu * kappa
    cdef double beta = (1.0 - mu) * kappa
    cdef double log_norm = lgamma(kappa) - lgamma(alpha) - lgamma(beta)

    for k in prange(M, schedule="static", nogil=True):
        log_p_het_k = _log_p_het_row(X, k, J)
        log_p_som_k = _log_p_som_row(X, logit_phi, k, J, alpha, beta, log_norm)
        log_pi_k = log_logistic(logit_pi[k])
        log1m_pi_k = log_logistic(-logit_pi[k])
        log_eta_num = log_pi_k + log_p_het_k
        log_eta_denom = logaddexp(log_eta_num, log1m_pi_k + log_p_som_k)
        eta_out[k] = exp(log_eta_num - log_eta_denom)
        _fill_gammas_row(X, logit_phi, gammas_out, k, J, alpha, beta, log_norm)


cpdef double kappa_Q(
        long[:, :, :] X, double[:, :] gammas,
        double mu, double kappa):
    """Evaluate the kappa M-step objective Q(kappa).

    Computes the expected log Beta-Binomial contribution to the complete-data
    log-likelihood, weighted by the somatic responsibilities (1 - gamma_jk),
    as in Eq. 13.

    Parameters
    ----------
    X : long[:, :, :]
        Read count array of shape (M, J, 2); X[k, j, 0] = ref reads,
        X[k, j, 1] = alt reads.
    gammas : double[:, :]
        Clone-level carrier responsibilities from the E-step, shape (M, J).
    mu : double
        Mean of the Beta-Binomial error distribution.
    kappa : double
        Concentration parameter to evaluate the objective at.

    Returns
    -------
    double
        Q(kappa) = sum_{k,j} (1 - gamma_kj)
                   * log_betabinom(a_kj, n_kj, mu*kappa, (1-mu)*kappa).
    """
    cdef int k, j, M = X.shape[0], J = X.shape[1]
    cdef double Q = 0.0, a, n
    cdef double alpha = mu * kappa
    cdef double beta = (1.0 - mu) * kappa
    cdef double log_norm = lgamma(kappa) - lgamma(alpha) - lgamma(beta)
    for k in prange(M, schedule="static", nogil=True):
        for j in range(J):
            a = X[k, j, 1]
            n = X[k, j, 0] + X[k, j, 1]
            Q += (1.0 - gammas[k, j]) * log_betabinom_prenorm(
                <long>a, <long>n, alpha, beta, log_norm
            )
    return Q


cpdef double kappa_score(
        long[:, :, :] X, double[:, :] gammas,
        double mu, double kappa):
    """Compute the score dQ/dkappa for use in Brent's method.

    Evaluates the first derivative of the kappa M-step objective with respect
    to kappa using digamma functions, as in Eq. 14. Passed as the bracketing
    function to scipy ``brentq``.

    Parameters
    ----------
    X : long[:, :, :]
        Read count array of shape (M, J, 2); X[k, j, 0] = ref reads,
        X[k, j, 1] = alt reads.
    gammas : double[:, :]
        Clone-level carrier responsibilities from the E-step, shape (M, J).
    mu : double
        Mean of the Beta-Binomial error distribution.
    kappa : double
        Concentration parameter at which to evaluate the score.

    Returns
    -------
    double
        dQ/dkappa summed over all sites and clones.

    Notes
    -----
    dQ/dkappa = sum_{k,j} (1 - gamma_kj) * [
        mu * psi(a + mu*kappa) + (1-mu) * psi(n-a + (1-mu)*kappa)
        - psi(n + kappa) - mu * psi(mu*kappa)
        - (1-mu) * psi((1-mu)*kappa) + psi(kappa)
    ]
    where psi denotes the digamma function.
    """
    cdef int k, j, M = X.shape[0], J = X.shape[1]
    cdef double score = 0.0, a, n, w
    for k in prange(M, schedule="static", nogil=True):
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
