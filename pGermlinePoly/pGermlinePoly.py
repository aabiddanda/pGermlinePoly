"""Inference and simulation of germline polymorphism in clonal sequencing data."""
import msprime
import numpy as np
from poly_utils import complete_loglik, geno_loglik, log_prior, logsumexp
from scipy.optimize import minimize
from scipy.stats import beta, binom, expon, norm, poisson, rv_histogram, uniform


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
            - lambdas (`np.array`): weight parameters for logistic priors.

        Returns:
            - post_k (`np.array`): posterior probability of site being germline polymorphic.

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
            - lambdas (`np.array`): weight parameters for logistic priors.

        Returns:
            - logll (`float`): log-likelihood of the model.

        """
        assert lambdas.size == self.A
        # run the complete log-likelihood helper function ...
        logll = complete_loglik(self.K, lambdas, self.Theta, self.X)
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

    def __init__(self, seq_len=1e7, n_clones=10):
        """Initialize the class for a simulation of clonal samples."""
        assert seq_len > 0
        assert n_clones > 1
        assert isinstance(n_clones, int)
        assert isinstance(seq_len, float) or isinstance(seq_len, int)
        self.seq_len = seq_len
        self.K = None
        self.J = n_clones
        self.genealogy = None

    def simulate_germline(
        self,
        afs=None,
        mean_coverage=15.0,
        var_coverage=5.0,
        mut_rate=1.2e-8,
        q=30,
        seed=42,
    ):
        """Simulate a new germline sample.

        Arguments:
            - afs (`np.array`): distribution of allele frequencys in the population (external).
            - mean_coverage (`float`): mean coverage of germline sample.
            - var_coverage (`float`): variance in coverage of germline sample.
            - mut_rate (`float`): rate of denovo mutations per-genome.
            - q (`float`): average quality of reads on phred-scale.
            - seed (`float`): random number seed.
        Returns:
            - ClonalSim object
        """
        assert mean_coverage > 0
        assert var_coverage > 0
        assert mut_rate > 0
        assert seed > 0
        np.random.seed(seed)
        # Simulate the total number of heterozygous sites
        if afs is None:
            # Draw from a uniform distribution ...
            ps = beta.rvs(1, 1, size=int(self.seq_len))
        else:
            # This is the case where we actually have an AFS ...
            assert afs.size > 10
            rv = rv_histogram(
                np.histogram(afs, bins=np.min([1000, afs.size / 20]).astype(np.int32))
            )
            ps = rv.rvs(size=int(self.seq_len / 1e3))
        # simulate genotypes under an HWE assumption
        gts = binom.rvs(2, p=ps)
        n_hets = np.sum(gts == 1)
        if n_hets == 0:
            raise ValueError("No heterozygotes simulated!")
        # simulate some denovo mutations
        denovo_muts = poisson.rvs(mu=self.seq_len * mut_rate)
        # Assign positions, afs categories
        tot_muts = n_hets + denovo_muts
        mut_pos = uniform.rvs(loc=0, scale=self.seq_len, size=tot_muts)
        mut_af = np.zeros(tot_muts)
        mut_af[:n_hets] = ps[gts == 1]
        mut_tot_reads = np.zeros(tot_muts, dtype=int)
        mut_alt_reads = np.zeros(tot_muts, dtype=int)
        mut_pl = np.zeros(shape=(tot_muts, 3))
        # Sample the total number of reads approximately from a normal distribution
        mut_tot_reads = np.round(
            norm.rvs(loc=mean_coverage, scale=np.sqrt(var_coverage), size=tot_muts)
        ).astype(int)
        mut_tot_reads[mut_tot_reads <= 0] = 0
        mut_alt_reads = binom.rvs(n=mut_tot_reads, p=0.5)
        for i, (a, t) in enumerate(zip(mut_alt_reads, mut_tot_reads)):
            # Estimate the genotyoe PL field based on this ...
            mut_pl[i, :] = geno_loglik(a, t, q=q)
        # Set all of the simulation object definitions for germline polymophism ...
        self.n_germline_poly = tot_muts
        self.n_denovo_muts = denovo_muts
        self.germline_muts = mut_pos
        self.germline_af = mut_af
        self.germline_tot_reads = mut_tot_reads
        self.germline_alt_reads = mut_alt_reads
        self.germline_pl = mut_pl

    def simulate_clone_genealogy(self, age=45, seed=42):
        """Simulate a number of clonal samples under a neutral bounded-coalescent model.

        Arguments:
            - age (`int`): the age of the individual at time of sampling.
            - seed (`int`): the random seed for simulating data
        Returns:
            - networkx object reflecting the joint genealogy simulated under a bounded coalescent model
        """
        assert age > 0.0
        assert self.J > 1
        assert seed > 0
        # This simulates a single locus genealogy for clones from a given age
        # NOTE: this is under Ne = 1.0 so we can rescale the branch-lengths accordingly ...
        ts = msprime.sim_ancestry(samples=self.J, ploidy=1, random_seed=42)
        self.genealogy = ts.at(0.0)

    def sim_somatic_mutations(
        self, age=45, mut_rate=5e-9, mean_coverage=15.0, var_coverage=5.0, q=30, seed=42
    ):
        """Simulate somatic mutations on branches of a latent somatic genealogy.

        Arguments:
            - age (`int`): the age of the individual in years
            - mut_rate (`float`): the somatic mutation rate in terms of /bp/year (note this is the diploid rate)
            - mean_coverage (`float`): mean coverage for clone.
            - var_coverage (`float`): variance in coverage for clone.
            - q (`int`): phred-scaled read-quality.
            - seed (`int`): random number seed.
        """
        assert self.genealogy is not None
        assert age > 0.0
        assert mut_rate > 0.0
        assert seed > 0
        assert mean_coverage > 0.0
        assert var_coverage > 0.0
        assert q > 0
        np.random.seed(seed)
        # Strong check that the appropriate number of leaves are available ...
        assert self.genealogy.num_samples(self.genealogy.root) == self.J
        # Obtain the height of the tree and set as the age
        # NOTE: this is a little improper as the somatic lineages
        #    may have coalesced well before the actual age of the sample
        g_height = self.genealogy.time(self.genealogy.root)
        scale_factor = age / g_height
        # Iterate through the branches of the genealogy
        n_somatic_mut = 0
        mut_pos = []
        mut_af = []
        mut_tot_reads = []
        mut_alt_reads = []
        for n in self.genealogy.nodes():
            # Get the branch-length and simulate the number of mutations on this branch
            bl = self.genealogy.branch_length(n)
            e_mut = mut_rate * bl * self.seq_len
            n_mut = poisson.rv(mu=e_mut)
            if n_mut > 0:
                n_somatic_mut += n_mut
                leaves = np.array([lv for lv in self.genealogy.leaves(n)])
                for _ in range(n_mut):
                    # Sample the position of the variant ...
                    cur_pos = uniform.rvs(loc=0, scale=self.seq_len)
                    # Sample total read-counts for the
                    cur_tot_reads = np.round(
                        norm.rvs(
                            loc=mean_coverage, scale=np.sqrt(var_coverage), size=self.J
                        )
                    ).astype(int)
                    cur_alt_reads = np.zeros(mut_tot_reads.size, dtype=int)
                    for lv in leaves:
                        # NOTE: in this simulation all somatic mutations are heterozygotes?
                        mut_alt_reads[lv] = binom.rvs(n=mut_tot_reads[lv], p=0.5)
                    mut_pos.append(cur_pos)
                    mut_af.append(0.0)
                    mut_tot_reads.append(cur_tot_reads)
                    mut_alt_reads.append(cur_alt_reads)
        mut_pos = np.array(mut_pos)
        mut_af = np.array(mut_af)
        mut_tot_reads = np.vstack(mut_tot_reads)
        mut_alt_reads = np.vstack(mut_alt_reads)
        self.n_somatic_mut = n_somatic_mut
        self.somatic_muts = mut_pos
        self.somatic_af = mut_af
        self.somatic_tot_reads = mut_tot_reads
        self.somatic_alt_reads = mut_alt_reads
        # If there are somatic mutations - estimate the pl field & add to the germline sample as ref ...
        if self.n_somatic_mut > 0:
            somatic_mut_pl = np.zeros(shape=(n_somatic_mut, self.J, 3))
            for i in range(n_somatic_mut):
                for j in range(self.J):
                    somatic_mut_pl[i, j, :] = geno_loglik(
                        self.somatic_alt_reads[i, j], self.somatic_tot_reads[i, j], q=q
                    )
            self.somatic_mut_pl = somatic_mut_pl

    def simulate_clonal_germline_muts(
        self, mean_coverage=15.0, var_coverage=5.0, q=30, seed=42
    ):
        """Simulate germline mutations in the clonal samples."""
        assert mean_coverage > 0
        assert var_coverage > 0
        assert q > 0
        assert seed > 0
        assert self.n_germline_poly > 0
        assert self.J > 1
        np.random.seed(seed)
        germline_clone_tot_reads = np.zeros(shape=(self.n_germline_poly, self.J))
        germline_clone_alt_reads = np.zeros(shape=(self.n_germline_poly, self.J))
        germline_clone_pl = np.zeros(shape=(self.n_germline_poly, self.J, 3))
        # Sample for each of
        for i in range(self.n_germline_poly):
            cur_tot_reads = np.round(
                norm.rvs(loc=mean_coverage, scale=np.sqrt(var_coverage), size=self.J)
            )
            cur_tot_reads[cur_tot_reads <= 0] = 0
            # NOTE: this could break due to
            cur_alt_reads = binom.rvs(n=cur_tot_reads, p=0.5)
            germline_clone_tot_reads[i, :] = cur_tot_reads
            germline_clone_alt_reads[i, :] = cur_alt_reads
            for j in range(self.J):
                germline_clone_pl[i, j, :] = geno_loglik(
                    germline_clone_alt_reads[i, j], germline_clone_tot_reads[i, j], q=q
                )
        # Store the clonal genotypes below ...
        self.germline_clone_tot_reads = germline_clone_tot_reads
        self.germline_clone_alt_reads = germline_clone_alt_reads
        self.germline_clone_pl = germline_clone_pl

    def simulate_germline_somatic_muts(
        self, mean_coverage=15.0, var_coverage=5.0, q=30, seed=42
    ):
        """Simulate the somatic mutations in the germline context."""
        assert self.n_somatic_mut >= 0
        assert mean_coverage > 0
        assert var_coverage > 0
        assert q > 0
        assert seed > 0
        np.random.seed(seed)
        somatic_tot_reads = np.zeros(self.n_somatic_mut, dtype=int)
        somatic_alt_reads = np.zeros(self.n_somatic_mut, dtype=int)
        somatic_pl = np.zeros(shape=(self.n_somatic_mut, 2))
        for i in range(self.n_somatic_mut):
            somatic_tot_reads[i] = np.round(
                norm.rvs(loc=mean_coverage, scale=np.sqrt(var_coverage))
            ).astype(int)
            somatic_alt_reads[i] = binom.rvs(n=somatic_tot_reads[i], p=0.0)
            somatic_pl[i, :] = geno_loglik(
                somatic_alt_reads[i], somatic_tot_reads[i], q=q
            )
        self.germline_somatic_tot_reads = somatic_tot_reads
        self.germline_somatic_alt_reads = somatic_alt_reads
        self.germline_somatic_pl = somatic_pl

    def write_vcf(self):
        """Write the VCF with clonal samples out."""
        pass
