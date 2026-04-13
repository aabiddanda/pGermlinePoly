import numpy as np
import pytest
from conftest import sim_read_counts, sim_annotations

from pGermlinePoly import ProbGermline


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


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_est_vaf(m, j, c, a):
    X = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
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
    X = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    vaf = np.array([X[i, :, 1].sum() / X[i, :, :].sum() for i in range(m)])
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf(naive=False)
    assert ~np.all(prob_germline.vaf == vaf)


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_est_vaf_CI(m, j, c, a):
    X = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
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
    X = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    llrs = prob_germline.loglik_ratio_het()
    assert llrs.size == prob_germline.M


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("a", [1, 5, 10])
def test_prior_poly(m, j, c, a):
    X = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
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
    X = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
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
    X = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
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
    X = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    A = sim_annotations(m=m, a=a, seed=m + a)
    prob_germline = ProbGermline(X=X, Theta=A)
    prob_germline.impute_anno()
    prob_germline.mle_vaf()
    lambdas_hat = prob_germline.naive_mle()
    lambdas = np.zeros(a)
    logll_null = prob_germline.complete_logll(lambdas=lambdas)
    logll_mle = prob_germline.complete_logll(lambdas=lambdas_hat)
    assert logll_mle > logll_null


# @pytest.mark.parametrize("m", [10, 50, 200])
# @pytest.mark.parametrize("j", [5, 50, 100])
# @pytest.mark.parametrize("c", [5, 10, 30, 50])
# @pytest.mark.parametrize("a", [1, 5, 10])
# def test_em_algo(m, j, c, a):
#     """Test a naive optimization of the weights for all SNPs."""
#     X = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
#     A = sim_annotations(m=m, a=a, seed=m + a)
#     prob_germline = ProbGermline(X=X, Theta=A)
#     prob_germline.impute_anno()
#     prob_germline.mle_vaf()
#     lambdas = np.zeros(a)
#     _, lambdas_hat = prob_germline.em_algo(lambdas=lambdas)
#     logll_null = prob_germline.complete_logll(lambdas=lambdas)
#     logll_mle = prob_germline.complete_logll(lambdas=lambdas_hat)
#     assert logll_mle >= logll_null
