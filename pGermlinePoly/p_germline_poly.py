import numpy as np
from poly_utils import *
from scipy.optimize import minimize


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
        return logll

    def incomplete_logll(self, gammas_k, lambdas=np.array([-1,-3])):
    		"""Compute the incomplete."""
    		# post_k = self.post_prob_het(lambdas=lambdas_prev)
    		assert post_k.size == self.K
    		logll = 0.0
    		for k in range(self.K):
    				x_k = np.sum(lambdas * anno[k, :])
    				pi_k = 1.0 / (1.0 + np.exp(-x_k))
    				logll += gammas_k[k]*(np.log(pi_k) + np.sum(X[k, :, 1:-1]))
    				logll += (1 - gammas_k[k])*(np.log(1.0 - pi_k) + np.sum(X[k, :, [0, -1]]))
  			return logll

  	def opt_lambdas(self, lambdas_prev=np.array([-1,-2])):
  			"""Optimize the lambda parameters in the incomplete log-likelihood for EM."""
  			post_k = self.post_prob_het(lambdas=lambdas_prev)
  			opt_res = minimize(
            lambda x: -self.incomplete_logll(
                gammas_k = post_k,
                lambdas=x
            )
            x0=lambdas_prev,
            method="L-BFGS-B",
            bounds=[[-100, 100] for k in range(self.A)],
            tol=1e-4,
            options={"disp": True, "ftol": 1e-4, "xtol": 1e-4},
        )
        lambda_hat = opt_res.x
        return lambda_hat

    def em_algo(self, lambdas=np.array([-1, -2]), delta_logll=1e-2):
        """EM-algorithm to estimate parameters for prior of germline polymorphism."""
        assert lambdas.size == self.A
        prev_lambdas = lambdas
        loglls = []
        loglls.append(self.incomplete_logll(lambdas=lambdas_prev, lambdas=lambdas_prev))
        cur_delta = 1e9
        while cur_delta >= delta_logll:
