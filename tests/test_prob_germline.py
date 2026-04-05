import numpy as np
import pytest

from pGermlinePoly import ProbGermline
# from pGermlinePoly.io import (
#     create_anno,
#     create_clonal_pl_matrix,
#     create_germline_anno_gl,
#     invert_pl,
# )

# The second variant should definitely have some contribution
X = np.array(
    [
        [[2, 3], [3, 4]],
        [[1, 0], [8, 8]],
        [[3, 3], [5, 5]],
    ],
    dtype="int",
)
Theta = np.array([[2.0, 0.0], [0.0, 0.1], [1.1, np.nan]], dtype="double")

# # A stronger example where the first annotation can matter quite a bit ...
# X2 = np.array(
#     [
#         [
#             invert_pl([10, 0, 5]),
#             invert_pl([4, 0, 5]),
#             invert_pl([10, 0, 20]),
#             invert_pl([5, 0, 20]),
#         ],
#         [
#             invert_pl([0, 5, 10]),
#             invert_pl([10, 0, 20]),
#             invert_pl([0, 5, 20]),
#             invert_pl([0, 2, 4]),
#         ],
#         [
#             invert_pl([3, 0, 7]),
#             invert_pl([3, 0, 7]),
#             invert_pl([4, 0, 12]),
#             invert_pl([3, 0, 4]),
#         ],
#     ],
#     dtype="double",
# )
# Theta2 = np.array([[5.0, 0.05], [0.0, 0.05], [5.0, 0.1]], dtype="double")


@pytest.mark.parametrize("m,j,a", [(10, 4, 2)])
def test_initialization(m, j, a):
    """Test that the intialization of the class will go well."""
    X = np.zeros(shape=(m, j, 2))
    A = np.zeros(shape=(m, a))
    print(X.shape, A.shape)
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


@pytest.mark.parametrize("X,A", [(X, Theta)])
def test_est_vaf(X, A):
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()


@pytest.mark.parametrize("X,A", [(X, Theta)])
def test_llr_het(X, A):
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    llrs = prob_germline.loglik_ratio_het()
    assert llrs.size == prob_germline.M


@pytest.mark.parametrize("X,A", [(X, Theta)])
def test_prior_poly(X, A):
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    pp = prob_germline.prior_poly(lambdas=np.zeros(A.shape[1]))
    assert np.all(pp >= 0.0) and np.all(pp <= 1.0)


@pytest.mark.parametrize("X,A", [(X, Theta)])
def test_posterior_prob_poly(X, A):
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    pp = prob_germline.post_prob_poly(lambdas=np.zeros(A.shape[1]))
    assert np.all(pp < 0.0)


# def test_post_prob_even():
#     """Ideally the posterior of these with a flat prior should be 0.5."""
#     X = np.array(
#         [
#             [invert_pl([0, 0, 0]), invert_pl([0, 0, 0])],
#             [invert_pl([0, 0, 0]), invert_pl([0, 0, 0])],
#         ],
#         dtype="double",
#     )
#     Theta = np.array([[0.0, 0.0], [0.0, 0.0]], dtype="double")
#     prob_germline = ProbGermline(X=X, Theta=Theta)
#     prob_germline.impute_anno()
#     post_k = prob_germline.post_prob_poly(npts=100)
#     assert post_k.size == prob_germline.K
#     assert np.all(np.isclose(np.exp(post_k), 0.5, atol=1e-2))


# def test_post_prob_poly():
#     """Simple simulation with just two points here."""
#     X = np.array(
#         [
#             [invert_pl([1, 0, 20]), invert_pl([3, 0, 2])],
#             [invert_pl([0, 20, 40]), invert_pl([0, 1, 2])],
#         ],
#         dtype="double",
#     )
#     Theta = np.array([[1.0, 1.0], [1.0, 1.0]], dtype="double")
#     prob_germline = ProbGermline(X=X, Theta=Theta)
#     prob_germline.impute_anno()
#     post_k = prob_germline.post_prob_poly()
#     assert post_k.size == prob_germline.K
#     # These should all be values less than 0.0 in log-space
#     assert np.all(post_k <= 0)
#     # The first value should be more likely to be a germline variant ...
#     assert post_k[0] > post_k[1]
#     assert np.exp(post_k[0]) > np.exp(post_k[1])


# def test_post_prob_poly_strong():
#     """Simple simulation with just two points here."""
#     X = np.array(
#         [
#             [invert_pl([20, 0, 20]), invert_pl([0, 10, 30])],
#             [invert_pl([20, 0, 40]), invert_pl([20, 0, 30])],
#             [invert_pl([10, 0, 40]), invert_pl([10, 0, 20])],
#         ],
#         dtype="double",
#     )
#     Theta = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype="double")
#     prob_germline = ProbGermline(X=X, Theta=Theta)
#     prob_germline.impute_anno()
#     a0_hat, mle_lambdas = prob_germline.naive_mle()
#     post_k = prob_germline.post_prob_poly(lambdas=mle_lambdas, a0=a0_hat)
#     assert post_k.size == prob_germline.K
#     # These should all be values less than 0.0 in log-space
#     assert np.all(post_k <= 0)
#     # The first value should be more likely to be a germline variant ...
#     assert post_k[1] > post_k[0]
#     assert post_k[2] > post_k[0]
#     assert np.exp(post_k[0]) > 0.8


# def test_complete_logll():
#     """Test the implementation of the complete log-likelihood for a very small test-case."""
#     prob_germline = ProbGermline(X=X, Theta=Theta)
#     prob_germline.impute_anno()
#     # The second log-likelihood should be a better fit since the first annotation should be more predictive of a true germline het...
#     logll1 = prob_germline.complete_logll(lambdas=np.array([2, 2], dtype="double"))
#     logll2 = prob_germline.complete_logll(lambdas=np.array([3, -1], dtype="double"))
#     assert logll2 >= logll1


# def test_naive_mle():
#     """Test the implementation of the complete log-likelihood for a very small test-case."""
#     prob_germline = ProbGermline(X=X, Theta=Theta)
#     prob_germline.impute_anno()
#     _, logll_p = prob_germline.mle_est_loglik()
#     a0_hat, mle_lambdas = prob_germline.naive_mle()
#     logll1 = prob_germline.complete_logll(lambdas=np.array([1.0, 1.0], dtype="double"))
#     logll2 = prob_germline.complete_logll(lambdas=mle_lambdas)
#     assert logll2 >= logll1


# def test_naive_mle2():
#     """Test the implementation of the complete log-likelihood for a very small test-case."""
#     prob_germline = ProbGermline(X=X2, Theta=Theta2)
#     prob_germline.impute_anno()
#     _, logll_p = prob_germline.mle_est_loglik()
#     a0_hat, mle_lambdas = prob_germline.naive_mle()
#     logll1 = prob_germline.complete_logll(lambdas=np.array([1.0, 1.0], dtype="double"))
#     logll2 = prob_germline.complete_logll(lambdas=mle_lambdas)
#     assert logll2 >= logll1
#     # There should be different signs for each of these predictions ...
#     assert mle_lambdas[0] > 0
#     assert mle_lambdas[1] < 0


# def test_loglik_ratio():
#     """Testing for sign of lambda estimate."""
#     prob_germline = ProbGermline(X=X2, Theta=Theta2)
#     prob_germline.impute_anno()
#     ll_ratio = prob_germline.loglik_ratio()
#     # The true somatic mutation below should be the main case here ...
#     assert ll_ratio[1] > ll_ratio[0]
#     assert ll_ratio[1] > ll_ratio[2]


# def test_vaf_est():
#     Theta = np.array([[0.0, 0.05], [0.0, 0.05], [0.0, 0.05]], dtype="double")
#     prob_germline = ProbGermline(X=X, Theta=Theta)
#     prob_germline.impute_anno()
#     mle_p, _, ci_mle_p = prob_germline.est_vaf_CI()
#     assert np.all(mle_p == ci_mle_p[:, 1])
#     assert np.all(ci_mle_p[:, 0] < ci_mle_p[:, 2])


# def test_naive_mle_from_vcf(tmp_path):
#     clone_sim = ClonalSim(seq_len=1e6, n_clones=10)
#     clone_sim.simulate_germline()
#     clone_sim.simulate_clone_genealogy(age=80)
#     clone_sim.sim_somatic_mutations(age=80, mut_rate=1e-6)
#     assert clone_sim.n_somatic_mut > 0
#     clone_sim.simulate_germline_somatic_muts()
#     assert np.any(clone_sim.germline_somatic_pl != 0)
#     clone_sim.simulate_clonal_germline_muts()
#     assert np.any(clone_sim.germline_clone_pl != 0)
#     # Setup the output VCF file and write it out
#     d = tmp_path / "em_vcf_test"
#     d.mkdir()
#     vcf_fp = d / "test.vcf"
#     clone_sim.write_vcf(out=vcf_fp)
#     # Reread the vcf file with the appropriate samples & create annotations ...
#     germline_samples = ["Agermline"]
#     clone_samples = [f"Aclone{i}" for i in range(10)]
#     germline_vcf = VCF(vcf_fp, samples=germline_samples, threads=1)
#     germline_anno = create_germline_anno_gl(germline_vcf)
#     clonal_vcf = VCF(vcf_fp, samples=clone_samples, threads=1)
#     clone_pl = create_clonal_pl_matrix(clonal_vcf)
#     anno_vcf = VCF(vcf_fp, samples=clone_samples, threads=1)
#     anno = create_anno(anno_vcf, annotations=["ExternalAF"])
#     # Make sure that the dimensions make sense here ...
#     full_anno = np.vstack([germline_anno, anno]).T
#     p_germline = ProbGermline(X=clone_pl, Theta=full_anno)
#     p_germline.impute_anno()
#     a0_hat, mle_lambdas = p_germline.naive_mle()
#     logll1 = p_germline.complete_logll(lambdas=np.array([0.0, 0.0], dtype="double"))
#     logll2 = p_germline.complete_logll(lambdas=mle_lambdas)
#     assert logll2 >= logll1
