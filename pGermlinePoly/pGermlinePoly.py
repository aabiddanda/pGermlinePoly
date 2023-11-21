"""Inference and simulation of germline polymorphism in clonal sequencing data."""
import numpy as np
from poly_utils import geno_loglik, log_prior, logsumexp
from scipy.optimize import minimize


class ProbGermline:
    """Class to estimate the posterior probability of germline polymorphism from somatic data."""

    def __init__(self, X, Theta):
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

    def impute_anno(self):
        """Impute annotations using the site-wise mean."""
        assert self.Theta is not None
        col_means = np.nanmean(self.Theta, axis=0)
        inds = np.where(np.isnan(self.Theta))
        self.Theta[inds] = np.take(col_means, inds[1])

    def post_prob_poly(self, lambdas=np.array([-1, -2])):
        """Posterior probability of being germline polymorphic.

        Arguments:
            - lambdas (`np.array`):

        Returns:
            - post_k (`np.array`):

        """
        assert lambdas.size == self.A
        assert np.all(~np.isnan(lambdas))
        post_k = np.zeros(self.K)
        for k in range(self.K):
            # Estimate the prior based on the weighted annotations
            pi_k = log_prior(lambdas, self.Theta[k, :])
            # Compute the posterior as an average across all the clones
            # NOTE: we assume that X contains the log-likelihood GL values...
            post_poly_k = np.log(pi_k) + np.sum(self.X[k, :, 1:-1])
            post_nonpoly_k = np.log(1.0 - pi_k) + np.sum(self.X[k, :, [0, -1]])
            post_k[k] = post_poly_k - logsumexp([post_poly_k, post_nonpoly_k])
        return post_k

    def complete_logll(self, lambdas=np.array([-1, -2])):
        """Compute the complete data log-likelihood.

        Arguments:
            - lambdas (`np.array`):

        """
        assert lambdas.size == self.A
        logll = 0.0
        for k in range(self.K):
            pi_k = log_prior(lambdas, self.Theta[k, :])
            # Compute the likelihood as a sum across sites
            logll += logsumexp(
                [
                    np.log(pi_k) + np.sum(self.X[k, :, 1:-1]),
                    np.log(1.0 - pi_k) + np.sum(self.X[k, :, [0, -1]]),
                ]
            )
        return logll

    def incomplete_logll(self, gammas_k, lambdas=np.array([-1, -3])):
        """Compute the incomplete-data log-likelihood for optimization.

        Note: this assumes that you have the posteriors pre-computed.

        """
        assert gammas_k.size == self.K
        logll = 0.0
        for k in range(self.K):
            pi_k = log_prior(lambdas, self.Theta[k])
            logll += gammas_k[k] * (np.log(pi_k) + np.sum(self.X[k, :, 1:-1]))
            logll += (1 - gammas_k[k]) * (
                np.log(1.0 - pi_k) + np.sum(self.X[k, :, [0, -1]])
            )
        return logll

    def opt_lambdas(self, gammas_k, algo="L-BFGS-B"):
        """Optimize the lambda parameter weights in the M-step of the EM-algorithm."""
        assert algo in ["L-BFGS-B", "Powell", "Nelder-Mead"]
        opt_res = minimize(
            lambda x: -self.incomplete_logll(gammas_k=gammas_k, lambdas=x),
            x0=[0 for _ in range(self.A)],
            method=algo,
            bounds=[(-100.0, 100.0) for k in range(self.A)],
            tol=1e-4,
            options={"disp": True, "ftol": 1e-4, "xtol": 1e-4},
        )
        lambda_hat = opt_res.x
        return lambda_hat

    def em_algo(self, lambdas=np.array([-1, -2]), delta_logll=1e-2):
        """EM-algorithm to estimate parameters for prior of germline polymorphism."""
        assert lambdas.size == self.A
        lambdas_prev = lambdas
        loglls = []
        loglls.append(self.complete_logll(lambdas=lambdas_prev))
        cur_delta = 1e9
        while cur_delta >= delta_logll:
            # E-step: estimate the expected probability
            gammas_k = self.post_prob_poly(lambdas=lambdas_prev)
            # M-step: maximize the parameters
            lambdas_hat = self.opt_lambdas(gammas_k=np.exp(gammas_k))
            loglls.append(self.complete_logll(lambdas=lambdas_hat))
            cur_delta = np.abs(loglls[-1] - loglls[-2])
            prev_lambdas = lambdas_hat
        return loglls, prev_lambdas


class ClonalSim:
    """A class for simulating clonal sequencing data."""

    def __init__(self, n_sites=50e6, n_clones=10):
        """Initialize the class for a simulation of clonal samples."""
        assert n_sites > 0
        assert n_clones > 0
        assert isinstance(n_clones, int)
        assert isinstance(n_sites, float) or isinstance(n_sites, int)
        self.K = n_sites
        self.J = n_clones
        self.genealogy = None

    def simulate_germline(
        self, afs=None, mean_coverage=15.0, var_coverage=5.0, mut_rate=1.2e-8
    ):
        """Simulate a new germline sample.

        Arguments:
            - afs (`np.array`): distribution of allele frequencys in the population (external).
            - mean_coverage (`float`): mean coverage of germline sample.
            - var_coverage (`float`): variance in coverage of germline sample.
            - mut_rate (`float`): rate of denovo mutations per-genome.

        Return:
            - gl_dict (`dict`): dictionary of mutations (position, (ref_reads, alt_reads), gls)).
        """
        assert mean_coverage > 0
        assert var_coverage > 0
        assert mut_rate > 0
        pass

    def simulate_clones(self, age=45):
        """Simulate a number of clonal samples under a neutral conditional-coalescent model.

        Arguments:
            - age (`int`): the age of the individual at time of sampling.
        """
        assert age > 0.0
        pass

    def sim_somatic_mutations(self, mut_rate=6e-6):
        """Simulate somatic mutations on branches of a somatic genealogy."""
        assert self.genealogy is not None
        assert mut_rate > 0.0
        pass

    def write_vcf(self):
        """Write the VCF with clonal samples out."""
        pass
