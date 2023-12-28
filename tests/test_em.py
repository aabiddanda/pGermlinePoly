import numpy as np
import pytest
from cyvcf2 import VCF
from poly_utils import logsumexp

from pGermlinePoly import ClonalSim, ProbGermline
from pGermlinePoly.io import (
    create_anno,
    create_clonal_pl_matrix,
    create_germline_anno_gl,
    invert_pl,
)

# The second variant should definitely have some kind of
X = np.array(
    [
        [invert_pl([10, 0, 5]), invert_pl([4, 0, 5])],
        [invert_pl([0, 5, 10]), invert_pl([5, 0, 10])],
        [invert_pl([3, 0, 7]), invert_pl([3, 0, 7])],
    ],
    dtype="double",
)
Theta = np.array([[2.0, 0.0], [0.0, 0.1], [1.1, np.nan]], dtype="double")


@pytest.mark.parametrize("k,j,a", [(10, 4, 2)])
def test_initialization(k, j, a):
    """Test that the intialization of the class will go well."""
    X = np.zeros(shape=(k, j, 3))
    A = np.zeros(shape=(k, a))
    prob_germline = ProbGermline(X=X, Theta=A)
    assert prob_germline.J == j
    assert prob_germline.K == k


def test_impute_anno():
    """Naive test that imputing annotations works."""
    X = np.zeros(shape=(10, 3, 3))
    Theta = np.zeros(shape=(10, 2))
    # Set a couple of nan values in there ...
    Theta[0, 0] = np.nan
    Theta[5, :] = np.nan
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    # Make sure we recover the nanmean...
    assert prob_germline.Theta[0, 0] == 0
    assert prob_germline.Theta[5, 0] == 0
    assert prob_germline.Theta[5, 1] == 0


def test_post_prob_even():
    """Ideally the posterior of these with a flat prior should be 0.5."""
    X = np.array(
        [
            [invert_pl([0, 0, 0]), invert_pl([0, 0, 0])],
            [invert_pl([0, 0, 0]), invert_pl([0, 0, 0])],
        ],
        dtype="double",
    )
    Theta = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="double")
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    post_k = prob_germline.post_prob_poly()
    assert post_k.size == prob_germline.K
    assert np.all(np.isclose(np.exp(post_k), 0.5))


def test_post_prob_poly():
    """Simple simulation with just two points here."""
    X = np.array(
        [
            [invert_pl([1, 0, 20]), invert_pl([3, 0, 2])],
            [invert_pl([0, 20, 40]), invert_pl([0, 1, 2])],
        ],
        dtype="double",
    )
    Theta = np.array([[1.0, 1.0], [1.0, 1.0]], dtype="double")
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    post_k = prob_germline.post_prob_poly()
    assert post_k.size == prob_germline.K
    # These should all be values less than 0.0 in log-space
    assert np.all(post_k <= 0)
    # The first value should be more likely to be a germline variant ...
    assert post_k[0] > post_k[1]
    assert np.exp(post_k[0]) > np.exp(post_k[1])


def test_complete_logll():
    """Test the implementation of the complete log-likelihood for a very small test-case."""
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    _, logll_p = prob_germline.mle_est_loglik()
    # The second log-likelihood should be a better fit since the first annotation should be more predictive of a true germline het...
    logll1 = prob_germline.complete_logll(
        lambdas=np.array([2, 2], dtype="double"), logll_p=logll_p
    )
    logll2 = prob_germline.complete_logll(
        lambdas=np.array([3, -1], dtype="double"), logll_p=logll_p
    )
    assert logll2 >= logll1
    assert logll2 < 0
    assert logll1 < 0


def test_naive_mle():
    """Test the implementation of the complete log-likelihood for a very small test-case."""
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    _, logll_p = prob_germline.mle_est_loglik()
    mle_lambdas = prob_germline.naive_mle(logll_p=logll_p)
    logll1 = prob_germline.complete_logll(lambdas=np.array([1.0, 1.0], dtype="double"))
    logll2 = prob_germline.complete_logll(lambdas=mle_lambdas)
    assert logll2 >= logll1


def test_loglik_ratio():
    """Testing for sign of lambda estimate."""
    X = np.array(
        [
            [
                invert_pl([10, 0, 5]),
                invert_pl([4, 0, 5]),
                invert_pl([10, 0, 20]),
                invert_pl([5, 0, 20]),
            ],
            [
                invert_pl([0, 5, 10]),
                invert_pl([10, 0, 20]),
                invert_pl([0, 5, 20]),
                invert_pl([0, 2, 4]),
            ],
            [
                invert_pl([3, 0, 7]),
                invert_pl([3, 0, 7]),
                invert_pl([4, 0, 12]),
                invert_pl([3, 0, 4]),
            ],
        ],
        dtype="double",
    )
    Theta = np.array([[5.0, 0.05], [0.0, 0.05], [5.0, 0.1]], dtype="double")
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    ll_ratio = prob_germline.loglik_ratio()
    # The true somatic mutation below should be the main case here ...
    assert ll_ratio[1] > ll_ratio[0]
    assert ll_ratio[1] > ll_ratio[2]


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
#     germline_anno = create_germline_anno(germline_vcf)
#     clonal_vcf = VCF(vcf_fp, samples=clone_samples, threads=1)
#     clone_pl = create_clonal_pl_matrix(clonal_vcf)
#     anno_vcf = VCF(vcf_fp, samples=clone_samples, threads=1)
#     anno = create_anno(anno_vcf, annotations=["ExternalAF", "AF"])
#     # Make sure that the dimensions make sense here ...
#     full_anno = np.vstack([germline_anno, anno]).T
#     p_germline = ProbGermline(X=clone_pl, Theta=full_anno)
#     p_germline.impute_anno()
#     mle_lambdas = p_germline.naive_mle()
#     logll1 = p_germline.complete_logll(
#         lambdas=np.array([0.0, 0.0, 0.0], dtype="double")
#     )
#     logll2 = p_germline.complete_logll(lambdas=mle_lambdas)
#     assert logll2 >= logll1
