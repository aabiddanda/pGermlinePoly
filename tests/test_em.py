import numpy as np
import pytest
from cyvcf2 import VCF
from poly_utils import logsumexp

from pGermlinePoly import ClonalSim, ProbGermline


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


def invert_pl(pl):
    """This is a test function to invert the PL field to be a scaled genotype log-likelihood."""
    p_gt = np.array(pl) / -10.0
    p_gt = np.nan_to_num(p_gt)
    return p_gt


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
    # The first value should be a more
    assert np.all(post_k <= 0)
    # The first value should be more likely to be a germline variant ...
    assert post_k[0] > post_k[1]
    assert np.exp(post_k[0]) > np.exp(post_k[1])


def test_complete_logll():
    """Test the implementation of the complete log-likelihood for a very small test-case."""
    X = np.array(
        [
            [invert_pl([1, 0, 20]), invert_pl([3, 0, 2])],
            [invert_pl([0, 20, 40]), invert_pl([0, 1, 2])],
        ],
        dtype="double",
    )
    Theta = np.array([[2.0, 1.0], [1.0, 1.0]], dtype="double")
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    # The second log-likelihood should be a better fit since the first annotation should be more predictive of a true germline het...
    logll1 = prob_germline.complete_logll(lambdas=np.array([2, 2], dtype="double"))
    logll2 = prob_germline.complete_logll(lambdas=np.array([-3, -1], dtype="double"))
    assert logll2 >= logll1


def test_incomplete_logll():
    """Test the incomplete log-likelihood for a small test case."""
    X = np.array(
        [
            [invert_pl([1, 0, 20]), invert_pl([3, 0, 2])],
            [invert_pl([0, 2, 4]), invert_pl([0, 1, 2])],
        ],
        dtype="double",
    )
    Theta = np.array([[2.0, 1.0], [1.0, 1.0]], dtype="double")
    gammas = np.array([0.5, 0.5], dtype="double")
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    logll1 = prob_germline.incomplete_logll(
        gammas_k=gammas, lambdas=np.array([2, 2], dtype="double")
    )
    logll2 = prob_germline.incomplete_logll(
        gammas_k=gammas, lambdas=np.array([-3, 0], dtype="double")
    )
    assert logll2 >= logll1


def test_em_algo():
    """Test out the full EM-algorithm."""
    X = np.array(
        [
            [invert_pl([1, 0, 20]), invert_pl([3, 0, 2])],
            [invert_pl([0, 2, 4]), invert_pl([0, 1, 2])],
        ],
        dtype="double",
    )
    Theta = np.array([[2.0, 1.0], [1.0, 1.0]], dtype="double")
    gammas = np.array([0.5, 0.5], dtype="double")
    prob_germline = ProbGermline(X=X, Theta=Theta)
    prob_germline.impute_anno()
    loglls, lambda_hats = prob_germline.em_algo(delta_logll=1e-4)
    # Basically we want to assure that it is increasing ...
    assert loglls.size > 1
    assert loglls[1] >= loglls[0]
