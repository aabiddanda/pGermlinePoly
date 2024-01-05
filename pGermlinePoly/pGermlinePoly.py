"""Inference and simulation of germline polymorphism in clonal sequencing data."""
import logging
import warnings

import msprime
import numpy as np
from poly_utils import (
    complete_loglik,
    d2_fun,
    geno_loglik,
    incomplete_loglik,
    log_prior,
    logaddexp,
    logsumexp,
    mle_est_loglik,
    posterior_poly,
    single_var_logll,
)
from scipy.optimize import minimize, minimize_scalar
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

    def __str__(self):
        """Return a string representation of the object."""
        return f"pGermlineObj ({self.K} sites; {self.J} clones; {self.A} annotations)"

    def impute_anno(self):
        """Impute annotations using the site-wise mean."""
        assert self.Theta is not None
        col_means = np.nanmean(self.Theta, axis=0)
        inds = np.where(np.isnan(self.Theta))
        self.Theta[inds] = np.take(col_means, inds[1])

    def mle_est_loglik(self):
        """Run the MLE estimation routine."""
        mle_p, logll_p = mle_est_loglik(J=self.J, K=self.K, X=self.X)
        return mle_p, logll_p

    def prior_poly(self, lambdas=np.array([0.0, 0.0], dtype="double")):
        """Prior probability of a germline polymorphism."""
        assert lambdas.size == self.A
        assert lambdas.ndim == 1
        pi_k = np.zeros(self.K)
        for k in range(self.K):
            pi_k[k] = log_prior(lambdas, self.Theta[k, :])
        return pi_k

    def post_prob_poly(
        self, lambdas=np.array([0.0, 0.0], dtype="double"), npts=20, a0=10.0
    ):
        """Posterior probability of being germline polymorphic.

        Arguments:
            - lambdas (`np.array`): weight parameters for logistic priors.

        Returns:
            - post_k (`np.array`): posterior probability (logged) of site being germline polymorphic.

        """
        assert lambdas.size == self.A
        assert np.all(~np.isnan(lambdas))
        assert a0 > 1.0

        post_k = np.zeros(self.K)
        for k in range(self.K):
            post_k[k] = posterior_poly(
                J=self.J,
                lambdas=lambdas,
                Theta=self.Theta[k, :],
                X=self.X[k, :, :],
                npts=npts,
                a0=a0,
            )
        return post_k

    def est_vaf_CI(self, h=1e-5):
        """Estimate the variant allele frequency from likelihoods across all the clonal data.

        Uses the fisher information to account for heterogeneous sequencing depth.
        """
        mle_p, logll_p = mle_est_loglik(K=self.K, J=self.J, X=self.X)
        ci_mle_p = np.zeros(shape=(self.K, 3))
        for k in range(self.K):
            ll = lambda p: single_var_logll(J=self.J, X=self.X[k, :, :], p=p)
            # Take the negative of the expectation of the second derivative ...
            fisher_I_inv = 1.0 / -d2_fun(ll, mle_p[k], h=h)
            ci_mle_p[k, 0] = mle_p[k] - 1.96 * np.sqrt(1.0 / self.J * fisher_I_inv)
            ci_mle_p[k, 1] = mle_p[k]
            ci_mle_p[k, 2] = mle_p[k] + 1.96 * np.sqrt(1.0 / self.J * fisher_I_inv)
        return mle_p, logll_p, ci_mle_p

    def loglik_ratio(self, logll_p=None):
        """Estimating the naive loglikelihood ratio."""
        if logll_p is None:
            _, logll_p = self.mle_est_loglik()
        logll_null = np.zeros(self.K)
        for k in range(self.K):
            logll_null[k] = single_var_logll(J=self.J, X=self.X[k, :, :], p=0.5)
        ll_ratio = -2 * (logll_null - logll_p)
        return ll_ratio

    def complete_logll(
        self, lambdas=np.array([0.0, 0.0], dtype="double"), a0=5.0, npts=20
    ):
        """Compute the complete data log-likelihood.

        Arguments:
            - lambdas (`np.array`): weight parameters for logistic priors.
            - a0 (`float`): the parameter for the beta distribution
            - npts (`int`): the number of points to evaluate the likelihood

        Returns:
            - logll (`float`): approximate log-likelihood of the model.

        """
        assert lambdas.size == self.A
        assert npts > 2
        assert a0 > 1.0
        logll = complete_loglik(
            K=self.K,
            J=self.J,
            lambdas=lambdas,
            Theta=self.Theta,
            X=self.X,
            a0=a0,
            npts=npts,
        )
        return logll

    def incomplete_logll(self, gammas_k, lambdas=np.array([0.0, 0.0], dtype="double")):
        """Compute the incomplete-data log-likelihood for optimization.

        Note: this assumes that you have the posteriors pre-computed.

        """
        assert gammas_k.size == self.K
        assert lambdas.size == self.A
        logll = incomplete_loglik(self.K, self.J, lambdas, gammas_k, self.Theta, self.X)
        return logll

    def naive_mle(self, algo="L-BFGS-B", npts=20, disp=False):
        """Naive optimization of the model log-likelihood.

        NOTE: this is not recommended for large models and largely is implemented for testing.

        Arguments:
            - algo (`string`): Type of optimization algorithm for likelihood.
            - npts (`int`): the number of points to evaluate the likelihood

        Returns:
            - a0_hat (`float`): approximate log-likelihood of the model.
            - lambda_hat (`np.array`): weight parameters for priors

        """
        assert algo in ["L-BFGS-B", "Powell", "Nelder-Mead"]
        opt_res = minimize(
            lambda x: -self.complete_logll(lambdas=x[1:], a0=x[0], npts=npts),
            x0=np.array([3.0] + [0.0 for _ in range(self.A)], dtype="double"),
            method=algo,
            bounds=[(0, 50)] + [(-20.0, 20.0) for _ in range(self.A)],
            tol=1e-8,
            options={"disp": disp},
        )
        est_hat = opt_res.x
        a0_hat = est_hat[0]
        lambda_hat = est_hat[1:]
        return a0_hat, lambda_hat

    def opt_lambdas(self, gammas_k, algo="L-BFGS-B"):
        """Optimize the lambda parameter weights in the M-step of the EM-algorithm."""
        assert algo in ["L-BFGS-B", "Powell", "Nelder-Mead"]
        opt_res = minimize(
            lambda x: -self.incomplete_logll(gammas_k=gammas_k, lambdas=x[1:], a0=x[0]),
            x0=np.array([3.0] + [0.0 for _ in range(self.A)], dtype="double"),
            method=algo,
            bounds=[(0, 50)] + [(-20.0, 20.0) for _ in range(self.A)],
            tol=1e-8,
            options={"disp": False},
        )
        a0_hat = opt_res.x[0]
        lambda_hat = opt_res.x[1:]
        return a0_hat, lambda_hat

    def em_algo(
        self,
        lambdas=np.array([0.0, 0.0], dtype="double"),
        a0=5.0,
        algo="L-BFGS-B",
        delta_logll=1e-6,
        log=True,
    ):
        """EM-algorithm to estimate parameters for prior of germline polymorphism.

        Arguments:
            - lambdas (`np.array`): starting weights for annotations
            - a0 (`float`): starting float

        """
        assert lambdas.size == self.A
        lambdas_prev = lambdas
        a0_prev = a0
        loglls = []
        loglls.append(self.complete_logll(lambdas=lambdas_prev))
        cur_delta = 1e9
        while cur_delta >= delta_logll:
            # E-step: estimate the expected probability using prev params
            gammas_k = self.post_prob_poly(lambdas=lambdas_prev)
            # M-step: maximize the parameters
            a0_hat, lambdas_hat = self.opt_lambdas(gammas_k=gammas_k, algo=algo)
            loglls.append(self.complete_logll(lambdas=lambdas_hat, a0=a0_hat))
            if log:
                logging.info(f"Log-likelihood {loglls[-1]}, Lambdas: {lambdas_hat}")
            if loglls[-1] >= loglls[-2]:
                warnings.warn("Incomplete log-likelihood is not increasing!")
            cur_delta = np.abs(loglls[-1] - loglls[-2])
            lambdas_prev = lambdas_hat
            a0_prev = a0_hat
        return np.array(loglls), a0_prev, lambdas_prev


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
        afs=[0.31699444395046117, 6.067159920986527],
        het_rate=1e-3,
        mean_coverage=15.0,
        var_coverage=5.0,
        mut_rate=1.2e-8,
        q=30,
        seed=42,
    ):
        """Simulate a new germline sample.

        Arguments:
            - afs (`np.array`): parameters of a beta distribution of allele frequencys in the population (external).
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
        # Estimate the number of heterozygotes per-bp as a Poisson random variable
        n_hets = poisson.rvs(mu=self.seq_len * het_rate)
        if n_hets == 0:
            raise ValueError("No heterozygotes simulated!")
        # Simulate the total number of heterozygous sites
        if afs is None:
            # Draw from a uniform beta prior + single-het observation posterior distribution
            ps = beta.rvs(1 + 1, 1 + 1, size=n_hets)
        elif len(afs) == 2:
            # Draw from the posterior distribution of single heterozygotes ...
            ps = beta.rvs(1 + afs[0], 1 + afs[1], size=n_hets)
        else:
            raise ValueError("Format / Type for AFS is incorrect!")
        # simulate some denovo mutations
        denovo_muts = poisson.rvs(mu=self.seq_len * mut_rate)
        # Assign positions, afs categories
        tot_muts = n_hets + denovo_muts
        mut_pos = uniform.rvs(loc=0, scale=self.seq_len, size=tot_muts)
        mut_af = np.zeros(tot_muts)
        mut_af[:n_hets] = ps
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
            # Estimate the genotype PL field based on this ...
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
            e_mut = bl * scale_factor * self.seq_len * mut_rate
            n_mut = poisson.rvs(mu=e_mut)
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
                    cur_alt_reads = np.zeros(self.J, dtype=int)
                    for lv in leaves:
                        # NOTE: in this simulation all somatic mutations are heterozygotes?
                        cur_alt_reads[lv] = binom.rvs(n=cur_tot_reads[lv], p=0.5)
                    mut_pos.append(cur_pos)
                    mut_af.append(0.0)
                    mut_tot_reads.append(cur_tot_reads)
                    mut_alt_reads.append(cur_alt_reads)
        mut_pos = np.array(mut_pos)
        mut_af = np.array(mut_af)
        if len(mut_tot_reads) > 0:
            mut_tot_reads = np.vstack(mut_tot_reads)
            mut_alt_reads = np.vstack(mut_alt_reads)
        else:
            mut_tot_reads = np.zeros(self.J)
            mut_alt_reads = np.zeros(self.J)
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
        """Simulate germline mutations in all of the clonal samples."""
        assert mean_coverage > 0
        assert var_coverage > 0
        assert q > 0
        assert seed > 0
        assert self.n_germline_poly > 0
        assert self.J > 1
        np.random.seed(seed)
        germline_clone_tot_reads = np.zeros(
            shape=(self.n_germline_poly, self.J), dtype=int
        )
        germline_clone_alt_reads = np.zeros(
            shape=(self.n_germline_poly, self.J), dtype=int
        )
        germline_clone_pl = np.zeros(shape=(self.n_germline_poly, self.J, 3))
        # Can we make these slightly faster?
        germline_clone_tot_reads = (
            np.round(
                norm.rvs(
                    loc=mean_coverage,
                    scale=np.sqrt(var_coverage),
                    size=int(self.J * self.n_germline_poly),
                )
            )
            .astype(int)
            .reshape((self.n_germline_poly, self.J))
        )
        germline_clone_tot_reads[germline_clone_tot_reads <= 0] = 0
        for i in range(self.n_germline_poly):
            germline_clone_alt_reads[i, :] = binom.rvs(
                n=germline_clone_tot_reads[i, :], p=0.5
            )
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
        somatic_pl = np.zeros(shape=(self.n_somatic_mut, 3))
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

    def create_gt_string(self, alt_reads=0, tot_reads=0, pl=np.array([0, 0, 0])):
        """Create a genotype-string."""
        assert pl.size > 1
        gt_str = "0/0"
        gt = 0
        an = 2
        if tot_reads >= 4 and alt_reads > 1:
            gt_str = "0/1"
            gt = 1
        if tot_reads < 2:
            gt_str = "./."
            an = 0
        ad_str = f"{tot_reads - alt_reads},{alt_reads}"
        dp_str = f"{tot_reads}"
        pl_str = ",".join([str(int(p)) for p in pl])
        gq = np.sort(pl)[1] - np.sort(pl)[0]
        gq_str = f"{int(gq)}"
        return f"{gt_str}:{ad_str}:{dp_str}:{gq_str}:{pl_str}", gt, an, tot_reads, gq

    def write_vcf(self, out=None):
        """Write the VCF with clonal samples out."""
        vcf_header = f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##ALT=<ID=NON_REF,Description="Represents any possible alternative allele not already represented at this location by REF and ALT">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth.">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency, for each ALT allele, in the same order as listed">
##INFO=<ID=ExternalAF,Number=A,Type=Float,Description="Global Allele Frequency, for each ALT allele, from external population reference">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth; some reads may have been filtered">
##INFO=<ID=SM,Number=1,Type=Integer,Description="Somatic mutation indicator.">
##contig=<ID=chr1,length={int(self.seq_len)}>
"""
        sample_header = (
            "\t".join(
                [
                    "#CHROM",
                    "POS",
                    "ID",
                    "REF",
                    "ALT",
                    "QUAL",
                    "FILTER",
                    "INFO",
                    "FORMAT",
                ]
                + ["Agermline"]
                + [f"Aclone{i}" for i in range(self.J)]
            )
            + "\n"
        )
        germline_var_strings = []
        cur_ref = "A"
        cur_alt = "T"
        cur_chrom = "chr1"
        # 1. Create strings for the germline mutations
        for i in range(self.n_germline_poly):
            cur_nalt = 0
            cur_nonmissing = 0
            cur_dp = []
            tot_gq = 0.0
            cur_pos = int(self.germline_muts[i])
            germline_alt_reads = self.germline_alt_reads[i]
            germline_tot_reads = self.germline_tot_reads[i]
            germline_pl = self.germline_pl[i, :]
            germline_str, gt, an, dp, gq = self.create_gt_string(
                germline_alt_reads, germline_tot_reads, germline_pl
            )
            cur_nalt += gt
            cur_nonmissing += an
            cur_dp.append(dp)
            external_af = self.germline_af[i]
            # Now creating the strings for germline variants in the somatic clones
            clone_gt_str = []
            for j in range(self.J):
                somatic_alt_reads = self.germline_clone_alt_reads[i, j]
                somatic_tot_reads = self.germline_clone_tot_reads[i, j]
                somatic_pl = self.germline_clone_pl[i, j, :]
                somatic_str, gt, an, dp, gq = self.create_gt_string(
                    somatic_alt_reads, somatic_tot_reads, somatic_pl
                )
                clone_gt_str.append(somatic_str)
                cur_nalt += gt
                cur_nonmissing += an
                tot_gq += gq
                cur_dp.append(dp)
            # Setting the info string here ...
            info_str = f"AC={cur_nalt};AF={cur_nalt / cur_nonmissing};AN={cur_nonmissing};DP={np.mean(cur_dp)};ExternalAF={external_af};SM=0"
            # Collapsing all of this into string output for this VCF record ...
            cur_var_str = (
                "\t".join(
                    [
                        cur_chrom,
                        str(cur_pos),
                        f"{cur_chrom}:{str(cur_pos)}:{cur_ref}:{cur_alt}",
                        cur_ref,
                        cur_alt,
                        str(tot_gq / (self.J + 1.0)),
                        "PASS",
                        info_str,
                        "GT:AD:DP:GQ:PL",
                        germline_str,
                    ]
                    + clone_gt_str
                )
                + "\n"
            )
            germline_var_strings.append(cur_var_str)
        # Create the same thing for the somatic mutations...
        somatic_var_strings = []
        for i in range(self.n_somatic_mut):
            cur_nalt = 0
            cur_nonmissing = 0
            tot_gq = 0.0
            cur_dp = []
            cur_pos = int(self.somatic_muts[i])
            germline_alt_reads = self.germline_somatic_alt_reads[i]
            germline_tot_reads = self.germline_somatic_tot_reads[i]
            germline_pl = self.germline_somatic_pl[i, :]
            germline_str, gt, an, dp, gq = self.create_gt_string(
                germline_alt_reads, germline_tot_reads, germline_pl
            )
            cur_nalt += gt
            cur_nonmissing += an
            tot_gq += gq
            cur_dp.append(dp)
            # Now creating the strings for germline variants in the somatic clones
            clone_gt_str = []
            external_af = self.somatic_af[i]
            for j in range(self.J):
                somatic_alt_reads = self.somatic_alt_reads[i, j]
                somatic_tot_reads = self.somatic_tot_reads[i, j]
                somatic_pl = self.somatic_mut_pl[i, j, :]
                somatic_str, gt, an, dp, gq = self.create_gt_string(
                    somatic_alt_reads, somatic_tot_reads, somatic_pl
                )
                clone_gt_str.append(somatic_str)
                cur_nalt += gt
                cur_nonmissing += an
                tot_gq += gq
                cur_dp.append(dp)
            # Setting the info string here ...
            info_str = f"AC={cur_nalt};AF={cur_nalt / cur_nonmissing};AN={cur_nonmissing};DP={np.mean(cur_dp)};ExternalAF={external_af};SM=1"
            # Collapsing all of this into string output for this VCF record ...
            cur_var_str = (
                "\t".join(
                    [
                        cur_chrom,
                        str(cur_pos),
                        f"{cur_chrom}:{cur_pos}:{cur_ref}:{cur_alt}",
                        cur_ref,
                        cur_alt,
                        str(tot_gq / (self.J + 1.0)),
                        "PASS",
                        info_str,
                        "GT:AD:DP:GQ:PL",
                        germline_str,
                    ]
                    + clone_gt_str
                )
                + "\n"
            )
            somatic_var_strings.append(cur_var_str)
        # 2. Create strings for the somatic variants
        with open(out, "wt") as out_stream:
            out_stream.write(vcf_header)
            out_stream.write(sample_header)
            for g_str in germline_var_strings:
                out_stream.write(g_str)
            for s_str in somatic_var_strings:
                out_stream.write(s_str)
