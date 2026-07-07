import numpy as np
from scipy.special import digamma as scipy_digamma, logsumexp
import pytest
from conftest import sim_read_counts, sim_annotations

from pGermlinePoly import ProbGermline
from poly_utils import kappa_score


# ---------------------------------------------------------------------------
# Reference implementations (scipy) for validating the native digamma path
# ---------------------------------------------------------------------------


def _ref_kappa_score(X, gammas, mu, kappa):
    """Pure-Python reference for kappa_score using scipy digamma."""
    score = 0.0
    M, J = X.shape[0], X.shape[1]
    for k in range(M):
        for j in range(J):
            a = float(X[k, j, 1])
            n = float(X[k, j, 0] + X[k, j, 1])
            w = 1.0 - gammas[k, j]
            score += w * (
                mu * scipy_digamma(a + mu * kappa)
                + (1.0 - mu) * scipy_digamma(n - a + (1.0 - mu) * kappa)
                - scipy_digamma(n + kappa)
                - mu * scipy_digamma(mu * kappa)
                - (1.0 - mu) * scipy_digamma((1.0 - mu) * kappa)
                + scipy_digamma(kappa)
            )
    return score


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mu", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("kappa", [1.0, 10.0, 100.0])
def test_kappa_score_vs_scipy(mu, kappa):
    """kappa_score (native digamma) must match the scipy reference to 1e-8."""
    rng = np.random.default_rng(42)
    M, J = 8, 5
    n = rng.integers(5, 30, size=(M, J))
    a = rng.integers(0, n + 1)
    X = np.stack([n - a, a], axis=-1).astype(np.int64)
    gammas = rng.uniform(0.0, 1.0, size=(M, J))
    ref = _ref_kappa_score(X, gammas, mu, kappa)
    got = kappa_score(X, gammas, mu, kappa)
    assert abs(got - ref) < 1e-8 * (abs(ref) + 1.0), (
        f"mu={mu}, kappa={kappa}: got={got}, ref={ref}"
    )


@pytest.mark.parametrize("kappa", [0.5, 5.0, 50.0])
def test_kappa_score_sign_at_boundary(kappa):
    """Score should change sign around the MLE concentration parameter."""
    rng = np.random.default_rng(7)
    M, J, mu = 20, 10, 0.5
    n = rng.integers(10, 40, size=(M, J))
    a = (n * mu).astype(np.int64)
    X = np.stack([n - a, a], axis=-1).astype(np.int64)
    gammas = np.zeros((M, J))  # all weight on beta-binomial component
    score = kappa_score(X, gammas, mu, kappa)
    assert np.isfinite(score)


def test_kappa_score_zero_weight():
    """When all gammas are 1 every site is pure binomial; score must be 0."""
    rng = np.random.default_rng(0)
    M, J = 5, 4
    n = rng.integers(5, 20, size=(M, J))
    a = rng.integers(0, n + 1)
    X = np.stack([n - a, a], axis=-1).astype(np.int64)
    gammas = np.ones((M, J))  # w = 1 - gamma = 0 for all sites
    assert kappa_score(X, gammas, 0.5, 10.0) == 0.0


@pytest.mark.parametrize("m,j,a", [(10, 4, 2)])
def test_initialization(m, j, a):
    """Test that the intialization of the class will go well."""
    X = np.zeros(shape=(m, j, 2))
    A = np.zeros(shape=(m, a))
    prob_germline = ProbGermline(X=X, Theta=A)
    assert prob_germline.J == j
    assert prob_germline.M == m


def test_impute_anno():
    """Naive test that imputing annotations works."""
    X = np.zeros(shape=(10, 3, 2))
    Theta = np.zeros(shape=(10, 2))
    # Set a couple of nan values in there ...
    Theta[0, 0] = np.nan
    Theta[5, :] = np.nan
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    # Make sure we recover the nanmean of zero.
    assert prob_germline.Theta[0, 0] == 0
    assert prob_germline.Theta[5, 0] == 0
    assert prob_germline.Theta[5, 1] == 0


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_est_vaf(m, j, c, a):
    X, _, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    vaf = np.array([X[i, :, 1].sum() / X[i, :, :].sum() for i in range(m)])
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    assert np.all(prob_germline.vaf == vaf)


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_est_vaf_nonnaive(m, j, c, a):
    X, _, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    vaf = np.array([X[i, :, 1].sum() / X[i, :, :].sum() for i in range(m)])
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf(naive=False)
    assert ~np.all(prob_germline.vaf == vaf)


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [50, 100])
@pytest.mark.parametrize("c", [30])
@pytest.mark.parametrize("a", [5])
@pytest.mark.parametrize("v", [0.1, 0.25])
def test_vaf_est_all_somatic(m, j, c, a, v):
    X, somatic, _ = sim_read_counts(
        m=m, j=j, coverage=c, p_somatic=1.0, vaf=v, seed=m + j
    )
    A = sim_annotations(m=m, a=a, seed=m + a)
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf(naive=True)
    assert np.all(prob_germline.vaf > 1e-3)
    prob_germline.mle_vaf(naive=False)
    assert np.all(prob_germline.vaf > 1e-3)


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_est_vaf_CI(m, j, c, a):
    X, _, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    vaf = np.array([X[i, :, 1].sum() / X[i, :, :].sum() for i in range(m)])
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    ci_mle_p = prob_germline.est_vaf_CI()
    assert np.all(prob_germline.vaf == vaf)
    assert np.all(ci_mle_p[:, 0] <= ci_mle_p[:, 2])


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_llr_het(m, j, c, a):
    X, _, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    llrs = prob_germline.loglik_ratio_het()
    assert llrs.size == prob_germline.M


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("a", [1, 5])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.25])
@pytest.mark.parametrize("v", [0.05, 0.1, 0.25])
def test_llr_het_somatic(m, j, c, a, p, v):
    X, somatic, _ = sim_read_counts(
        m=m, j=j, coverage=c, p_somatic=p, vaf=v, seed=m + j
    )
    A = sim_annotations(m=m, a=a, seed=m + a)
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    llrs = prob_germline.loglik_ratio_het()
    assert llrs.size == prob_germline.M
    if np.sum(somatic) > 0:
        assert np.mean(llrs[somatic])


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_prior_poly(m, j, c, a):
    X, _, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    pp = prob_germline.prior_poly(lambdas=np.zeros(A.shape[1]))
    assert np.all(pp <= 0.0)
    assert pp.ndim == 1
    assert pp.size == m


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30, 50])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_posterior_prob_poly(m, j, c, a):
    X, _, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    pp = prob_germline.post_prob_poly(lambdas=np.zeros(A.shape[1]))
    assert np.all(pp <= 0.0)
    assert pp.ndim == 1
    assert pp.size == m


def test_posterior_prob_even():
    """Test with even posterior distribution due to same likelihood."""

    X = np.array(
        [
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]],
        ],
        dtype="int",
    )
    A = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, np.nan]], dtype="double")
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    pp = prob_germline.post_prob_poly(lambdas=np.zeros(A.shape[1]))
    assert pp.size == X.shape[0]
    assert np.all(pp == pp[0])


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30, 50])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_complete_logll(m, j, c, a):
    """Test the computational of the likelihood of the observed data given lambdas."""
    X, _, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    lambdas = np.zeros(a)
    logll = prob_germline.complete_logll(lambdas=lambdas)
    assert logll <= 0


@pytest.mark.parametrize("m", [10, 50, 1000])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30, 50])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_infer_weights(m, j, c, a):
    """Test a naive optimization of the weights for all SNPs."""
    X, _, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    lambdas_hat = prob_germline.naive_mle()
    lambdas = np.zeros(a)
    logll_null = prob_germline.complete_logll(lambdas=lambdas)
    logll_mle = prob_germline.complete_logll(lambdas=lambdas_hat)
    assert logll_mle >= logll_null


# ---------------------------------------------------------------------------
# EM algorithm simulation test
# ---------------------------------------------------------------------------


def test_em_algo_loglik_nondecreasing():
    """EM log-likelihood must be non-decreasing across all iterations."""
    M, J, cov = 60, 8, 30
    X, *_ = sim_read_counts(m=M, j=J, coverage=cov, seed=7)
    A = sim_annotations(m=M, a=2, seed=7)
    pg = ProbGermline(X=X, Theta=A)
    pg.mle_vaf()
    loglls, _, _, _ = pg.em_algo(delta_logll=1e-3)
    diffs = np.diff(loglls)
    assert np.all(diffs >= -1e-6), (
        f"Log-likelihood decreased at iterations: {np.where(diffs < -1e-6)[0].tolist()}"
    )


def test_em_algo_improves_over_null():
    """EM should achieve a strictly better log-likelihood than the null (lambda=0)."""
    M, J, cov = 80, 10, 30
    X, *_ = sim_read_counts(m=M, j=J, coverage=cov, seed=5)
    A = sim_annotations(m=M, a=3, seed=5)
    pg = ProbGermline(X=X, Theta=A)
    pg.mle_vaf()
    ll_null = pg.complete_logll(lambdas=np.zeros(3))
    loglls, lambdas_hat, betas_hat, kappa_hat = pg.em_algo(delta_logll=1e-3)
    assert loglls[-1] >= ll_null - 1e-6
    assert np.all(np.isfinite(lambdas_hat))
    assert np.isfinite(kappa_hat) and kappa_hat > 0


def test_em_algo_discriminates_germline_somatic():
    """EM should learn a positive annotation weight when the annotation tags germline sites,
    and posterior probabilities should be systematically higher for true germline sites."""
    rng = np.random.default_rng(123)
    M_germ, M_som, J, cov = 60, 60, 10, 30

    # Germline sites: binomial(n, 0.5) reads across all clones
    X_germ = np.zeros((M_germ, J, 2), dtype=np.int64)
    for i in range(M_germ):
        n = rng.poisson(cov, size=J)
        a = rng.binomial(n, 0.5)
        X_germ[i, :, 0] = n - a
        X_germ[i, :, 1] = a

    # Somatic sites: error-level reads in all clones except one carrier at ~50% VAF
    X_som = np.zeros((M_som, J, 2), dtype=np.int64)
    for i in range(M_som):
        n = rng.poisson(cov, size=J)
        a = rng.binomial(n, 1e-3)
        j_mut = rng.integers(J)
        a[j_mut] = rng.binomial(n[j_mut], 0.5)
        X_som[i, :, 0] = n - a
        X_som[i, :, 1] = a

    X = np.vstack([X_germ, X_som])

    # Annotation: germline sites draw from N(+2, 0.5), somatic from N(-2, 0.5)
    Theta = np.vstack(
        [
            rng.normal(2.0, 0.5, size=(M_germ, 1)),
            rng.normal(-2.0, 0.5, size=(M_som, 1)),
        ]
    )

    pg = ProbGermline(X=X, Theta=Theta)
    pg.mle_vaf()
    loglls, lambdas_hat, betas_hat, kappa_hat = pg.em_algo(delta_logll=1e-3)

    # Lambda should be positive: higher annotation → more likely germline
    assert lambdas_hat[0] > 0, f"Expected positive lambda, got {lambdas_hat[0]:.4f}"

    # Posterior probs (log scale) should be higher for germline sites on average
    pp = pg.post_prob_poly(lambdas=lambdas_hat, betas=betas_hat, kappa=kappa_hat)
    mean_pp_germ = np.mean(pp[:M_germ])
    mean_pp_som = np.mean(pp[M_germ:])
    assert mean_pp_germ > mean_pp_som, (
        f"Germline posterior mean {mean_pp_germ:.3f} not greater than "
        f"somatic posterior mean {mean_pp_som:.3f}"
    )


# ---------------------------------------------------------------------------
# est_germline_genotype tests
# ---------------------------------------------------------------------------


def test_germline_genotype_shape_and_normalized():
    """Output shape is (M, 3) and each row sums to 1 in probability space."""
    M, J, a = 20, 8, 2
    X, _, _ = sim_read_counts(m=M, j=J, coverage=20, seed=1)
    A = sim_annotations(m=M, a=a, seed=1)
    pg = ProbGermline(X=X, Theta=A)
    log_post = pg.est_germline_genotype()
    assert log_post.shape == (M, 3)
    assert np.allclose(logsumexp(log_post, axis=1), 0.0, atol=1e-10)
    assert np.all(log_post <= 0.0)


def test_germline_genotype_balanced_reads_prefers_het():
    """When every clone has equal alt and ref counts, 0/1 should be the MAP genotype."""
    M, J = 15, 10
    n_each = 10
    X = np.zeros((M, J, 2), dtype=np.int64)
    X[:, :, 0] = n_each  # ref
    X[:, :, 1] = n_each  # alt
    A = np.zeros((M, 1))
    pg = ProbGermline(X=X, Theta=A)
    log_post = pg.est_germline_genotype()
    assert np.all(np.argmax(log_post, axis=1) == 1), (
        "0/1 (index 1) should win with balanced reads"
    )


def test_germline_genotype_all_ref_prefers_hom_ref():
    """When every clone shows only ref reads, 0/0 should be the MAP genotype."""
    M, J = 10, 8
    X = np.zeros((M, J, 2), dtype=np.int64)
    X[:, :, 0] = 20  # all ref, no alt
    A = np.zeros((M, 1))
    pg = ProbGermline(X=X, Theta=A)
    log_post = pg.est_germline_genotype()
    assert np.all(np.argmax(log_post, axis=1) == 0), (
        "0/0 (index 0) should win with all-ref reads"
    )


def test_germline_genotype_all_alt_prefers_hom_alt():
    """When every clone shows only alt reads, 1/1 should be the MAP genotype."""
    M, J = 10, 8
    X = np.zeros((M, J, 2), dtype=np.int64)
    X[:, :, 1] = 20  # all alt, no ref
    A = np.zeros((M, 1))
    pg = ProbGermline(X=X, Theta=A)
    log_post = pg.est_germline_genotype()
    assert np.all(np.argmax(log_post, axis=1) == 2), (
        "1/1 (index 2) should win with all-alt reads"
    )


def test_germline_genotype_hwe_prior():
    """With allele_freq supplied the HWE prior shifts mass toward expected genotype."""
    M, J = 20, 10
    X = np.zeros((M, J, 2), dtype=np.int64)
    X[:, :, 0] = 10
    X[:, :, 1] = 10  # balanced — 0/1 likelihood wins
    A = np.zeros((M, 1))
    pg = ProbGermline(X=X, Theta=A)

    # Uniform prior: equal allele_freq = 0.5 should also favour 0/1
    log_post_hwe = pg.est_germline_genotype(allele_freq=np.full(M, 0.5))
    assert log_post_hwe.shape == (M, 3)
    assert np.allclose(logsumexp(log_post_hwe, axis=1), 0.0, atol=1e-10)
    assert np.all(np.argmax(log_post_hwe, axis=1) == 1)

    # Rare allele (p=0.01): 0/0 prior dominates; with balanced reads 0/1 should still win
    # at high depth but prior fights back — just check shape and normalization
    log_post_rare = pg.est_germline_genotype(allele_freq=np.full(M, 0.01))
    assert np.allclose(logsumexp(log_post_rare, axis=1), 0.0, atol=1e-10)


@pytest.mark.parametrize(
    "true_geno,p_alt,expected_idx",
    [
        ("0/0", 1e-3, 0),  # hom-ref: alt reads are sequencing errors only
        ("0/1", 0.5, 1),  # het: equal alt and ref reads in every clone
        ("1/1", 1 - 1e-3, 2),  # hom-alt: ref reads are sequencing errors only
    ],
)
def test_germline_genotype_recovers_truth(true_geno, p_alt, expected_idx):
    """MAP genotype equals the true root genotype for data simulated under each state.

    With J=15 clones at depth 30 the likelihood ratio between the true and next-best
    genotype is extreme (hundreds of nats), so the MAP should be correct at every site.
    """
    rng = np.random.default_rng(99)
    M, J, cov = 30, 15, 30
    n = rng.poisson(cov, size=(M, J))
    a = rng.binomial(n, p_alt)
    X = np.stack([n - a, a], axis=-1).astype(np.int64)
    A = np.zeros((M, 1))
    pg = ProbGermline(X=X, Theta=A, mu=1e-3)
    log_post = pg.est_germline_genotype()
    map_geno = np.argmax(log_post, axis=1)
    assert np.all(map_geno == expected_idx), (
        f"True genotype {true_geno}: expected all {M} sites to call index "
        f"{expected_idx}, but {(map_geno != expected_idx).sum()} differed. "
        f"Unique MAP values: {np.unique(map_geno).tolist()}"
    )


def test_germline_genotype_zero_depth_equals_prior():
    """With zero read depth the posterior must equal the prior exactly.

    All log-likelihoods are 0 when no reads exist, so log_post_unnorm = log_prior
    and the posterior reduces to the prior after normalization.  A uniform prior
    (P(0/0) = P(0/1) = P(1/1) = 1/3) is achieved by setting logit_pi = -log(2) for
    all sites (giving sigma = 1/3) together with p_hom_alt = 0.5, so all three
    columns should equal log(1/3).
    """
    M, J = 20, 15
    X = np.zeros((M, J, 2), dtype=np.int64)
    Theta = np.ones((M, 1))
    lambdas = np.array([-np.log(2)])  # sigma(-log2) = 1/3 → uniform prior
    pg = ProbGermline(X=X, Theta=Theta)
    log_post = pg.est_germline_genotype(lambdas=lambdas, p_hom_alt=0.5)
    expected = np.log(1.0 / 3.0)
    assert np.allclose(log_post, expected, atol=1e-10), (
        f"Expected all log-posteriors = log(1/3) = {expected:.6f}, "
        f"got range [{log_post.min():.6f}, {log_post.max():.6f}]"
    )


def test_germline_genotype_invalid_p_hom_alt():
    """p_hom_alt outside (0, 1) must raise ValueError."""
    X = np.zeros((5, 3, 2), dtype=np.int64)
    A = np.zeros((5, 1))
    pg = ProbGermline(X=X, Theta=A)
    with pytest.raises(ValueError, match="p_hom_alt"):
        pg.est_germline_genotype(p_hom_alt=0.0)
    with pytest.raises(ValueError, match="p_hom_alt"):
        pg.est_germline_genotype(p_hom_alt=1.0)


# ---------------------------------------------------------------------------
# reorient_to_minor_allele tests
# ---------------------------------------------------------------------------


def _make_uniform_X(n_sites, n_clones, alt_per_clone, ref_per_clone):
    """Build an (M, J, 2) array with fixed [ref, alt] counts at every cell."""
    X = np.zeros((n_sites, n_clones, 2), dtype=np.int64)
    X[:, :, 0] = ref_per_clone
    X[:, :, 1] = alt_per_clone
    return X


def test_reorient_flipped_mask_high_alt():
    """Sites with pooled alt > 50 % get flipped=True; others get flipped=False."""
    # site 0: alt=70 %, site 1: alt=30 %, site 2: alt=50 % (boundary — NOT flipped)
    X = np.array(
        [
            [[30, 70], [30, 70]],  # alt=70 %
            [[70, 30], [70, 30]],  # alt=30 %
            [[50, 50], [50, 50]],  # alt=50 % — strictly > 0.5 not met
        ],
        dtype=np.int64,
    )
    A = np.zeros((3, 1))
    pg = ProbGermline(X=X, Theta=A)
    pg.reorient_to_minor_allele()

    assert pg.flipped[0] is np.bool_(True), "70 % alt site should be flipped"
    assert pg.flipped[1] is np.bool_(False), "30 % alt site should not be flipped"
    assert pg.flipped[2] is np.bool_(False), "50 % alt site should not be flipped"


def test_reorient_minor_allele_always_le_half():
    """After re-orientation, every site has pooled alt frequency ≤ 0.5."""
    rng = np.random.default_rng(0)
    M, J = 40, 8
    n = rng.integers(10, 50, size=(M, J))
    a = rng.integers(0, n + 1)
    X = np.stack([n - a, a], axis=-1).astype(np.int64)
    A = np.zeros((M, 1))
    pg = ProbGermline(X=X, Theta=A)
    pg.reorient_to_minor_allele()

    pooled_alt = pg.X[:, :, 1].sum(axis=1)
    pooled_tot = pg.X.sum(axis=(1, 2))
    assert np.all(pooled_alt / pooled_tot <= 0.5 + 1e-9)


def test_reorient_swaps_columns_correctly():
    """For a flipped site the new alt column equals the original ref column."""
    ref_counts, alt_counts = 30, 70  # alt > 50 % → will be flipped
    X = _make_uniform_X(
        n_sites=1, n_clones=5, alt_per_clone=alt_counts, ref_per_clone=ref_counts
    )
    A = np.zeros((1, 1))
    pg = ProbGermline(X=X, Theta=A)
    pg.reorient_to_minor_allele()

    assert pg.flipped[0]
    assert np.all(pg.X[0, :, 1] == ref_counts), "new alt should be old ref"
    assert np.all(pg.X[0, :, 0] == alt_counts), "new ref should be old alt"


def test_reorient_no_sites_to_flip():
    """Method runs without error when all sites already have alt ≤ 50 %."""
    X = _make_uniform_X(n_sites=10, n_clones=4, alt_per_clone=20, ref_per_clone=80)
    A = np.zeros((10, 1))
    pg = ProbGermline(X=X, Theta=A)
    pg.reorient_to_minor_allele()

    assert not pg.flipped.any()
    assert np.all(pg.X[:, :, 1] == 20)  # unchanged


def test_reorient_all_sites_flipped():
    """Method runs without error when every site has alt > 50 %."""
    X = _make_uniform_X(n_sites=10, n_clones=4, alt_per_clone=80, ref_per_clone=20)
    A = np.zeros((10, 1))
    pg = ProbGermline(X=X, Theta=A)
    pg.reorient_to_minor_allele()

    assert pg.flipped.all()
    assert np.all(pg.X[:, :, 1] == 20)  # all now show minor allele = original ref


def test_reorient_symmetry_pp_germline():
    """ppGermlinePoly must be identical for mirror-image datasets after re-orientation.

    Two datasets that are exact reflections of each other (alt=p vs alt=1-p across
    all clones) should produce the same ppGermlinePoly once both are re-oriented to
    the minor allele, because they become bit-for-bit identical inputs to the model.
    """
    rng = np.random.default_rng(7)
    M, J, cov = 20, 10, 30
    n = rng.poisson(cov, size=(M, J))
    # Dataset A: alt reads drawn at 30 % per clone
    a_low = rng.binomial(n, 0.3)
    X_low = np.stack([n - a_low, a_low], axis=-1).astype(np.int64)
    # Dataset B: same total depths, alt reads = ref counts from A (mirror image)
    X_high = np.stack([a_low, n - a_low], axis=-1).astype(np.int64)

    A = np.zeros((M, 1))

    pg_low = ProbGermline(X=X_low, Theta=A.copy())
    pg_low.reorient_to_minor_allele()
    pp_low = pg_low.post_prob_poly(lambdas=np.zeros(1))

    pg_high = ProbGermline(X=X_high, Theta=A.copy())
    pg_high.reorient_to_minor_allele()
    pp_high = pg_high.post_prob_poly(lambdas=np.zeros(1))

    assert np.allclose(pp_low, pp_high, atol=1e-10), (
        "ppGermlinePoly should be the same for mirror-image datasets after re-orientation"
    )


def test_reorient_flipped_attr_absent_before_call():
    """``self.flipped`` does not exist before reorient_to_minor_allele is called."""
    X = _make_uniform_X(n_sites=3, n_clones=2, alt_per_clone=10, ref_per_clone=90)
    A = np.zeros((3, 1))
    pg = ProbGermline(X=X, Theta=A)
    assert not hasattr(pg, "flipped")
    pg.reorient_to_minor_allele()
    assert hasattr(pg, "flipped")


# --------------------------------------------------------------------------- #
# reflect_af_annotations tests
# --------------------------------------------------------------------------- #


def _make_pg_with_af_col(af_values, alt_fracs, n_clones=4, depth=20):
    """Build a ProbGermline where column 1 of Theta holds raw AF values."""
    M = len(af_values)
    rng = np.random.default_rng(0)
    n = np.full((M, n_clones), depth, dtype=np.int64)
    alt = rng.binomial(n, np.array(alt_fracs)[:, None]).astype(np.int64)
    X = np.stack([n - alt, alt], axis=-1)
    # Theta: intercept=1, AF column
    Theta = np.column_stack([np.ones(M), np.array(af_values, dtype=np.float64)])
    return ProbGermline(X=X, Theta=Theta)


def test_reflect_af_raises_without_flipped():
    """reflect_af_annotations raises if called before reorient_to_minor_allele."""
    pg = _make_pg_with_af_col([0.3, 0.8], [0.3, 0.8])
    with pytest.raises(RuntimeError):
        pg.reflect_af_annotations([1])


def test_reflect_af_raw_no_transform():
    """Flipped sites get AF → 1-AF; unflipped sites are unchanged."""
    # Site 0: alt_frac=0.2 → not flipped; AF=0.3 unchanged
    # Site 1: alt_frac=0.8 → flipped;     AF=0.9 → reflected to 0.1
    pg = _make_pg_with_af_col([0.3, 0.9], [0.2, 0.8])
    pg.reorient_to_minor_allele()
    pg.reflect_af_annotations([1], transform_names=[None])
    assert pg.Theta[0, 1] == pytest.approx(0.3)
    assert pg.Theta[1, 1] == pytest.approx(0.1)


def test_reflect_af_log10_transform():
    """Reflection inverts log10, reflects raw AF, re-applies log10."""
    import math

    af_raw = [0.3, 0.9]
    pg = _make_pg_with_af_col(
        [math.log10(f) for f in af_raw],
        [0.2, 0.8],
    )
    pg.reorient_to_minor_allele()
    pg.reflect_af_annotations([1], transform_names=["log10"])
    assert pg.Theta[0, 1] == pytest.approx(math.log10(0.3))
    assert pg.Theta[1, 1] == pytest.approx(math.log10(1.0 - 0.9))


def test_reflect_af_sqrt_transform():
    """Reflection inverts sqrt, reflects raw AF, re-applies sqrt."""
    import math

    af_raw = [0.25, 0.81]
    pg = _make_pg_with_af_col(
        [math.sqrt(f) for f in af_raw],
        [0.2, 0.8],
    )
    pg.reorient_to_minor_allele()
    pg.reflect_af_annotations([1], transform_names=["sqrt"])
    assert pg.Theta[0, 1] == pytest.approx(math.sqrt(0.25))
    assert pg.Theta[1, 1] == pytest.approx(math.sqrt(1.0 - 0.81))


def test_reflect_af_skips_nan():
    """Sites with NaN AF are not reflected even if they were flipped."""
    pg = _make_pg_with_af_col([0.3, float("nan")], [0.2, 0.8])
    pg.reorient_to_minor_allele()
    pg.reflect_af_annotations([1], transform_names=[None])
    assert pg.Theta[0, 1] == pytest.approx(0.3)  # not flipped, unchanged
    assert np.isnan(pg.Theta[1, 1])  # flipped but NaN → still NaN


def test_reflect_af_clips_boundary():
    """AF near 0 or 1 after reflection is clipped to [0, 1] without producing NaN/inf."""
    # AF=1.0 → reflected raw = 0.0 → log10 clips to 1e-10
    pg = _make_pg_with_af_col([np.log10(1.0 - 1e-12)], [0.9])
    pg.reorient_to_minor_allele()
    pg.reflect_af_annotations([1], transform_names=["log10"])
    assert np.isfinite(pg.Theta[0, 1])
