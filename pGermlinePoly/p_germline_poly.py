import numpy as np
from poly_utils import *


class ProbGermline:
    """Class to estimate the posterior probability of germline."""

    def __init__(X, Theta):
        """Initialize the class.

        Arguments:
          - X (`np.array`): The genotype-likelihoods of each possible model (log-spaced ...)
          - Theta (`np.array`): The K x A matrix of annotations we are using.
        """

        assert X.ndim == 3
        self.K, self.J, _ = X.shape
        self.X = X
        assert Theta.ndim == 2
        K, self.A = Theta.shape
        assert K == self.K
        self.Theta = Theta

    def post_prob_het(self, lambdas=np.array([-1, -2])):
        """Posterior probability of being polymorphic in the germline."""
        assert lambdas.size == self.A
        pi_k = np.zeros(self.K)
        post_k = np.zeros(self.K)
        for k in range(self.K):
            # Estimate the prior based on the weighted annotations
            x_k = np.sum(lambdas * anno[k, :])
            pi_k = 1.0 / (1.0 + np.exp(-x_k))
            # Compute the posterior as an average across all the clones
            post_poly_k = np.log(pi_k) + np.sum(X[k, :, 1:-1])
            post_nonpoly_k = np.log(1 - pi_k) + np.sum(X[k, :, [0, -1]])
            post_k[k] = post_poly_k - logsumexp([post_poly_k, post_nonpoly_k])
        return post_k

    def complete_logll(self, lambdas=np.array([-1, -2])):
        """Compute the complete data log-likelihood."""
        assert lambdas.size == self.A
        pi_k = np.zeros(self.K)
        logll = 0.0
        for k in range(self.K):
            # Estimate the pik probabilities based on the held annotations
            x_k = np.sum(lambdas * anno[k, :])
            pi_k = 1.0 / (1.0 + np.exp(-x_k))
            # Compute the likelihood as a sum across sites
            logll += logsumexp(
                [
                    np.log(pi_k) + np.sum(X[k, :, 1:-1]),
                    np.log(1 - pi_k) + np.sum(X[k, :, [0, -1]]),
                ]
            )
        return post_k

    def em_algo(self, lambdas=np.array([-1, -2]), delta_logll=1e-2):
        """EM-algorithm to estimate parameters for prior of germline polymorphism."""
        assert lambdas.size == self.A
        prev_lambdas = lambdas
        loglls = []
        loglls.append()
        cur_delta = 1e9
        while cur_delta >= delta_logll:

            pass
