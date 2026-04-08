import numpy as np
import pytest

from pGermlinePoly import ProbGermline

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
