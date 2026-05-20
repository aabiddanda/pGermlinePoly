"""Inference and simulation of germline polymorphism in clonal sequencing data."""

import logging
import msprime
import numpy as np
import warnings

from poly_utils import (
    loglik_ratio,
    var_loglik,
    geno_loglik,
    log_posterior_germline,
    observed_loglik_site,
    e_step_all,
    kappa_score,
    sum_log_betabinom,
)
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.stats import beta, binom, chi2, norm, poisson, uniform

logger = logging.getLogger(__name__)


class ProbGermline:
    """Compute the posterior probability of germline polymorphism from clonal sequencing data.

    Implements an EM algorithm that jointly estimates logistic annotation
    weights (lambda, beta) and a Beta-Binomial error concentration (kappa)
    to discriminate germline heterozygotes from somatic variants.

    Parameters
    ----------
    X : numpy.ndarray
        Read-count array of shape (M, J, 2). ``X[:, :, 0]`` are reference
        read counts and ``X[:, :, 1]`` are alternative read counts.
        M = number of sites, J = number of clones.
    Theta : numpy.ndarray
        Site-level annotation matrix of shape (M, L).
    Phi : numpy.ndarray, optional
        Clone-level annotation array of shape (M, J, B). When provided,
        clone-specific beta weights are estimated in addition to lambda.
        Default is None.
    kappa : float, optional
        Initial concentration parameter for the Beta-Binomial error prior.
        Default is 100.0.
    mu : float, optional
        Fixed mean sequencing error rate used in the Beta-Binomial model.
        Default is 1e-3.

    Attributes
    ----------
    M : int
        Number of sites.
    J : int
        Number of clones.
    L : int
        Number of site-level annotations.
    B : int
        Number of clone-level annotations (0 if Phi is None).
    kappa : float
        Current (or estimated) concentration parameter.
    mu : float
        Fixed mean sequencing error rate.
    vaf : numpy.ndarray or None
        Per-site MLE variant allele frequencies, shape (M,). Set by
        :meth:`mle_vaf`.
    """

    def __init__(self, X, Theta, Phi=None, kappa=100.0, mu=1e-3):
        assert X.ndim == 3
        self.M, self.J, _ = X.shape
        self.X = np.ascontiguousarray(X, dtype=np.int64)
        assert Theta.ndim == 2
        M, self.L = Theta.shape
        assert M == self.M
        self.Theta = Theta
        self.A = self.L  # backward-compat alias

        if Phi is not None:
            assert Phi.ndim == 3
            assert Phi.shape[0] == self.M and Phi.shape[1] == self.J
            self.Phi = Phi
            self.B = Phi.shape[2]
            self.Phi_bar = Phi.mean(axis=1)  # M x B clone-average
        else:
            self.Phi = None
            self.B = 0
            self.Phi_bar = None

        self.kappa = float(kappa)
        self.mu = float(mu)
        self.vaf = None
        self.logl_vaf = None

    def __str__(self):
        """Return a brief string representation of the object."""
        return f"pGermlineObj ({self.M} sites {self.J} clones {self.A} annotations)"

    def impute_anno(self):
        """Impute missing annotation values using column-wise means.

        Replaces NaN entries in ``self.Theta`` in-place with the mean of the
        corresponding annotation column, computed over all non-missing sites.
        """
        assert self.Theta is not None
        col_means = np.nanmean(self.Theta, axis=0)
        inds = np.where(np.isnan(self.Theta))
        self.Theta[inds] = np.take(col_means, inds[1])

    def mle_vaf(self, naive=True, eps=1e-3, **kwargs):
        """Estimate the per-site MLE variant allele frequency from pooled clone reads.

        Results are stored in ``self.vaf`` (shape (M,)) and ``self.logl_vaf``
        (shape (M,)).

        Parameters
        ----------
        naive : bool, optional
            If True (default), uses the empirical proportion alt/(alt+ref).
            If False, optimises :func:`~poly_utils.var_loglik` via
            ``scipy.optimize.minimize_scalar``.
        eps : float, optional
            Sequencing error rate used in the likelihood. Default is 1e-3.
        **kwargs
            Forwarded to :func:`~poly_utils.var_loglik`.
        """
        mle_p = np.zeros(self.M)
        logll_p = np.zeros(self.M)
        if naive:
            for i in range(self.M):
                ax, rx = self.X[i, :, 1].sum(), self.X[i, :, 0].sum()
                mle_p[i] = ax / (ax + rx)
                logll_p[i] = var_loglik(ax, rx, f=mle_p[i], eps=eps)
        else:
            for i in range(self.M):
                ax, rx = self.X[i, :, 1].sum(), self.X[i, :, 0].sum()
                opt_res = minimize_scalar(
                    lambda p: -var_loglik(ax, rx, f=p, eps=eps),
                    bounds=(0.0, 1.0),
                )
                mle_p[i] = opt_res.x
                logll_p[i] = var_loglik(ax, rx, f=mle_p[i], eps=eps)
        self.vaf = mle_p
        self.logl_vaf = logll_p

    def loglik_ratio_het(self, **kwargs):
        """Compute the likelihood ratio statistic for each site.

        Returns 2*(LL_somatic - LL_het). Under the null hypothesis that the
        site is a germline heterozygote this is asymptotically chi-squared
        with 1 degree of freedom.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`mle_vaf` (if VAF has not yet been computed)
            and to :func:`~poly_utils.loglik_ratio`.

        Returns
        -------
        numpy.ndarray
            LRT statistics of shape (M,).
        """
        if self.vaf is None:
            self.mle_vaf(**kwargs)
        llr = np.zeros(self.M)
        for i in range(self.M):
            llr[i] = loglik_ratio(
                ax=self.X[i, :, 1], rx=self.X[i, :, 0], alpha=2 * self.vaf[i], **kwargs
            )
        return llr

    def _compute_logit_phi(self, lambdas, betas):
        """Compute logit carrier probabilities for all sites and clones.

        logit(phi_jk) = Theta_k @ lambdas + Phi_jk @ betas.

        Parameters
        ----------
        lambdas : numpy.ndarray
            Site-level annotation weights, shape (L,).
        betas : numpy.ndarray or None
            Clone-level annotation weights, shape (B,). If None or empty,
            only the site term is used.

        Returns
        -------
        numpy.ndarray
            Contiguous float64 array of shape (M, J).
        """
        site = self.Theta @ lambdas  # (M,)
        if self.B > 0 and betas is not None and betas.size > 0:
            clone = np.einsum("mjb,b->mj", self.Phi, betas)  # (M, J)
            logit = site[:, None] + clone
        else:
            logit = np.broadcast_to(site[:, None], (self.M, self.J)).copy()
        return np.ascontiguousarray(logit, dtype=np.float64)

    def _compute_logit_pi(self, lambdas, betas):
        """Compute logit prior germline probabilities for all sites.

        logit(pi_k) = Theta_k @ lambdas + Phi_bar_k @ betas, where Phi_bar
        is the clone-average of the clone-level annotations.

        Parameters
        ----------
        lambdas : numpy.ndarray
            Site-level annotation weights, shape (L,).
        betas : numpy.ndarray or None
            Clone-level annotation weights, shape (B,).

        Returns
        -------
        numpy.ndarray
            Contiguous float64 array of shape (M,).
        """
        logit = self.Theta @ lambdas  # (M,)
        if self.B > 0 and betas is not None and betas.size > 0:
            logit = logit + self.Phi_bar @ betas
        return np.ascontiguousarray(logit, dtype=np.float64)

    def prior_poly(self, lambdas=np.array([0.0, 0.0], dtype="double")):
        """Compute the log prior probability of germline heterozygosity.

        Evaluates log sigma(Theta @ lambdas) for all M sites under the
        logistic model.

        Parameters
        ----------
        lambdas : numpy.ndarray, optional
            Site-level annotation weights, shape (L,). Default is zeros.

        Returns
        -------
        numpy.ndarray
            Log prior probabilities of shape (M,), in (-inf, 0].
        """
        assert lambdas.size == self.L
        assert lambdas.ndim == 1
        logit_pi = self._compute_logit_pi(lambdas, np.zeros(self.B))
        return -np.log1p(np.exp(-logit_pi))  # log sigmoid — shape (M,)

    def post_prob_poly(
        self,
        lambdas=np.array([0.0, 0.0], dtype="double"),
        betas=None,
        kappa=None,
        **kwargs,
    ):
        """Compute the log posterior probability of germline heterozygosity for all sites.

        Evaluates log P(z_k = het | A_k, R_k) for each of the M sites
        using the current or supplied model parameters, as in Eq. 9.

        Parameters
        ----------
        lambdas : numpy.ndarray, optional
            Site-level annotation weights, shape (L,). Default is zeros.
        betas : numpy.ndarray or None, optional
            Clone-level annotation weights, shape (B,). None uses zeros.
        kappa : float or None, optional
            Beta-Binomial concentration parameter. None uses ``self.kappa``.
        **kwargs
            Accepted for interface compatibility; currently unused.

        Returns
        -------
        numpy.ndarray
            Log posterior probabilities of shape (M,), in (-inf, 0].
        """
        assert lambdas.size == self.L
        assert np.all(~np.isnan(lambdas))
        if betas is None:
            betas = np.zeros(self.B)
        if kappa is None:
            kappa = self.kappa

        logit_phi = self._compute_logit_phi(lambdas, betas)  # (M, J)
        logit_pi = self._compute_logit_pi(lambdas, betas)  # (M,)
        post_k = np.zeros(self.M)
        for k in range(self.M):
            post_k[k] = log_posterior_germline(
                ax=self.X[k, :, 1],
                rx=self.X[k, :, 0],
                logit_phi=logit_phi[k],
                logit_pi=logit_pi[k],
                mu=self.mu,
                kappa=kappa,
            )
        return post_k

    def est_vaf_CI(self, alpha=0.05, df=1, **kwargs):
        """Estimate per-site profile-likelihood confidence intervals for the VAF.

        Uses the Wilks approximation: the CI is the set of VAF values p for
        which 2*(LL_mle - LL_p) < chi2(1-alpha, df). Bounds are found via
        ``scipy.optimize.brentq``.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the CI. Default is 0.05 (95% CI).
        df : int, optional
            Degrees of freedom for the chi-squared threshold. Default is 1.
        **kwargs
            Forwarded to :func:`~poly_utils.var_loglik`.

        Returns
        -------
        numpy.ndarray
            Array of shape (M, 3) with columns [lower_CI, MLE_VAF, upper_CI].
        """
        assert (alpha > 0) and (alpha < 1)
        assert df > 0
        if self.vaf is None:
            self.mle_vaf()
        ci_mle_p = np.zeros(shape=(self.M, 3))
        qval = chi2.ppf(1.0 - alpha, df=df)
        for i, v in enumerate(self.vaf):
            ax, rx = self.X[i, :, 1].sum(), self.X[i, :, 0].sum()
            wilks = lambda p: (
                2
                * (
                    var_loglik(ax, rx, f=v, **kwargs)
                    - var_loglik(ax, rx, f=p, **kwargs)
                )
            )  # noqa
            try:
                lower_CI = brentq(lambda x: wilks(x) - qval, 1e-6, v)
            except ValueError:
                lower_CI = 0.0
            try:
                upper_CI = brentq(lambda x: wilks(x) - qval, v, 1.0)
            except ValueError:
                upper_CI = 1.0
            ci_mle_p[i, 0] = lower_CI
            ci_mle_p[i, 1] = v
            ci_mle_p[i, 2] = upper_CI
        return ci_mle_p

    def complete_logll(
        self,
        lambdas=np.array([0.0, 0.0], dtype="double"),
        betas=None,
        kappa=None,
        **kwargs,
    ):
        """Compute the observed data log-likelihood summed over all M sites.

        Evaluates sum_k log P(A_k, R_k) by marginalizing the latent class
        over each site.

        Parameters
        ----------
        lambdas : numpy.ndarray, optional
            Site-level annotation weights, shape (L,). Default is zeros.
        betas : numpy.ndarray or None, optional
            Clone-level annotation weights, shape (B,). None uses zeros.
        kappa : float or None, optional
            Beta-Binomial concentration parameter. None uses ``self.kappa``.
        **kwargs
            Accepted for interface compatibility; currently unused.

        Returns
        -------
        float
            sum_k log P(A_k, R_k | lambdas, betas, kappa).
        """
        assert lambdas.size == self.L
        if betas is None:
            betas = np.zeros(self.B)
        if kappa is None:
            kappa = self.kappa
        logit_phi = self._compute_logit_phi(lambdas, betas)  # (M, J)
        logit_pi = self._compute_logit_pi(lambdas, betas)  # (M,)
        logll = 0.0
        for k in range(self.M):
            logll += observed_loglik_site(
                ax=self.X[k, :, 1],
                rx=self.X[k, :, 0],
                logit_phi=logit_phi[k],
                logit_pi=logit_pi[k],
                mu=self.mu,
                kappa=kappa,
            )
        return logll

    def naive_mle(self, algo="L-BFGS-B", **kwargs):
        """Direct MLE of site-level annotation weights with betas fixed at zero.

        Maximises the observed log-likelihood via ``scipy.optimize.minimize``.

        Parameters
        ----------
        algo : str, optional
            Scipy minimisation algorithm. One of ``"L-BFGS-B"``,
            ``"Powell"``, or ``"Nelder-Mead"``. Default is ``"L-BFGS-B"``.
        **kwargs
            Forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        numpy.ndarray
            MLE site-level annotation weights, shape (L,).
        """
        assert algo in ["L-BFGS-B", "Powell", "Nelder-Mead"]
        opt_res = minimize(
            lambda x: -self.complete_logll(lambdas=x),
            x0=np.zeros(self.L),
            method=algo,
            bounds=[(-20, 20) for _ in range(self.L)],
            **kwargs,
        )
        return opt_res.x

    # ------------------------------------------------------------------
    # EM algorithm (current model)
    # ------------------------------------------------------------------

    def _e_step(self, lambdas, betas, kappa):
        """Run the E-step and return site-level and clone-level responsibilities.

        Parameters
        ----------
        lambdas : numpy.ndarray
            Site-level annotation weights, shape (L,).
        betas : numpy.ndarray
            Clone-level annotation weights, shape (B,).
        kappa : float
            Beta-Binomial concentration parameter.

        Returns
        -------
        eta : numpy.ndarray
            Site-level posterior probabilities P(z_k = het | data), shape (M,).
        gammas : numpy.ndarray
            Clone-level carrier responsibilities, shape (M, J).
        """
        logit_phi = self._compute_logit_phi(lambdas, betas)  # (M, J) contiguous float64
        logit_pi = self._compute_logit_pi(lambdas, betas)  # (M,)
        eta = np.zeros(self.M, dtype=np.float64)
        gammas = np.zeros((self.M, self.J), dtype=np.float64)
        e_step_all(self.X, logit_phi, logit_pi, self.mu, kappa, eta, gammas)
        return eta, gammas

    def _m_step_lambda_beta(self, eta, gammas, lambdas0, betas0, algo="L-BFGS-B"):
        """Run the M-step to update annotation weights via weighted logistic regression.

        Minimises the negative Q function with respect to (lambda, beta) using
        ``scipy.optimize.minimize``, as in Eq. 12.

        Parameters
        ----------
        eta : numpy.ndarray
            Site-level responsibilities from the E-step, shape (M,).
        gammas : numpy.ndarray
            Clone-level responsibilities from the E-step, shape (M, J).
        lambdas0 : numpy.ndarray
            Initial site-level annotation weights, shape (L,).
        betas0 : numpy.ndarray
            Initial clone-level annotation weights, shape (B,).
        algo : str, optional
            Scipy minimisation algorithm. Default is ``"L-BFGS-B"``.

        Returns
        -------
        lambdas : numpy.ndarray
            Updated site-level annotation weights, shape (L,).
        betas : numpy.ndarray
            Updated clone-level annotation weights, shape (B,).
        """
        params0 = np.concatenate([lambdas0, betas0])
        L, B = self.L, self.B
        Theta = self.Theta  # (M, L)
        Phi = self.Phi  # (M, J, B) or None
        Phi_bar = self.Phi_bar  # (M, B) or None

        def neg_Q(params):
            lam = params[:L]
            bet = params[L:]

            logit_pi = Theta @ lam
            if B > 0:
                logit_pi = logit_pi + Phi_bar @ bet
            # phi per clone
            site_part = (Theta @ lam)[:, None]  # (M, 1)
            if B > 0:
                clone_part = np.einsum("mjb,b->mj", Phi, bet)  # (M, J)
                logit_phi = site_part + clone_part
            else:
                logit_phi = np.broadcast_to(site_part, (self.M, self.J))

            log_pi = -np.log1p(np.exp(-logit_pi))
            log1m_pi = -np.log1p(np.exp(logit_pi))
            log_phi = -np.log1p(np.exp(-logit_phi))
            log1m_phi = -np.log1p(np.exp(logit_phi))

            site_term = np.dot(eta, log_pi) + np.dot(1.0 - eta, log1m_pi)
            clone_term = (gammas * log_phi + (1.0 - gammas) * log1m_phi).sum()
            return -(site_term + clone_term)

        bounds = [(-20.0, 20.0)] * (L + B)
        opt = minimize(
            neg_Q,
            params0,
            method=algo,
            bounds=bounds,
            tol=1e-8,
            options={"disp": False},
        )
        return opt.x[:L], opt.x[L:]

    def _m_step_kappa(self, gammas):
        """Run the M-step to update kappa via Brent's method on the score function.

        The score dQ/dkappa is evaluated by the Cython
        :func:`~poly_utils.kappa_score` function. Brent's method is applied
        over [1e-2, 1e6]; if no sign change is found the current kappa is
        kept, as in Eqs. 13-14.

        Parameters
        ----------
        gammas : numpy.ndarray
            Clone-level responsibilities from the E-step, shape (M, J).

        Returns
        -------
        float
            Updated concentration parameter kappa.
        """
        score_fn = lambda k: kappa_score(self.X, gammas, self.mu, k)
        lo, hi = 1e-2, 1e6
        try:
            if score_fn(lo) * score_fn(hi) >= 0:
                return self.kappa  # no sign change — keep current value
            kappa_hat = brentq(score_fn, lo, hi, xtol=1e-6, rtol=1e-6)
        except ValueError:
            kappa_hat = self.kappa
        return float(kappa_hat)

    def em_algo(
        self,
        lambdas=None,
        betas=None,
        kappa=None,
        algo="L-BFGS-B",
        delta_logll=1e-4,
        **kwargs,
    ):
        """Run the EM algorithm to jointly estimate (lambda, beta, kappa).

        Iterates E-step / M-step until the absolute change in observed
        log-likelihood falls below ``delta_logll``, as described in
        Eqs. 10-14.

        Parameters
        ----------
        lambdas : numpy.ndarray or None, optional
            Initial site-level annotation weights, shape (L,). None uses zeros.
        betas : numpy.ndarray or None, optional
            Initial clone-level annotation weights, shape (B,). None uses zeros.
        kappa : float or None, optional
            Initial Beta-Binomial concentration. None uses ``self.kappa``.
        algo : str, optional
            Scipy optimizer for the (lambda, beta) M-step. One of
            ``"L-BFGS-B"``, ``"Powell"``, or ``"Nelder-Mead"``.
            Default is ``"L-BFGS-B"``.
        delta_logll : float, optional
            Convergence threshold on the absolute change in observed
            log-likelihood between successive EM iterations. Default is 1e-4.
        **kwargs
            Currently unused; accepted for forward compatibility.

        Returns
        -------
        loglls : numpy.ndarray
            Observed log-likelihood trace, length = number of iterations + 1.
        lambdas_hat : numpy.ndarray
            Estimated site-level annotation weights, shape (L,).
        betas_hat : numpy.ndarray
            Estimated clone-level annotation weights, shape (B,).
        kappa_hat : float
            Estimated Beta-Binomial concentration parameter.
        """
        if lambdas is None:
            lambdas = np.zeros(self.L)
        if betas is None:
            betas = np.zeros(self.B)
        if kappa is None:
            kappa = self.kappa

        assert lambdas.size == self.L
        assert betas.size == self.B

        loglls = [self.complete_logll(lambdas=lambdas, betas=betas, kappa=kappa)]
        cur_delta = 1e9
        iteration = 0
        logger.info("EM iter %d  loglik=%.6f", iteration, loglls[-1])

        while cur_delta >= delta_logll:
            iteration += 1

            # E-step
            eta, gammas = self._e_step(lambdas, betas, kappa)

            # M-step: logistic weights
            lambdas, betas = self._m_step_lambda_beta(
                eta, gammas, lambdas, betas, algo=algo
            )

            # M-step: kappa (Brent on Cython score function)
            kappa = self._m_step_kappa(gammas)

            new_ll = self.complete_logll(lambdas=lambdas, betas=betas, kappa=kappa)
            if new_ll < loglls[-1] - 1e-8:
                warnings.warn("Observed log-likelihood decreased in EM iteration.")
            loglls.append(new_ll)
            cur_delta = abs(loglls[-1] - loglls[-2])
            logger.info(
                "EM iter %d  loglik=%.6f  delta=%.2e  kappa=%.4f",
                iteration,
                new_ll,
                cur_delta,
                kappa,
            )

        logger.info(
            "EM converged in %d iterations  loglik=%.6f  kappa=%.4f",
            iteration,
            loglls[-1],
            kappa,
        )
        self.kappa = kappa
        return np.array(loglls), lambdas, betas, kappa


class MutectLOD:
    """Compute per-site LOD scores following the Mutect2 / Williams et al. model.

    Parameters
    ----------
    X : numpy.ndarray
        Read-count array of shape (M, J, 2). Only biallelic variants are
        supported (``X.shape[2]`` must equal 2).

    Attributes
    ----------
    M : int
        Number of sites.
    J : int
        Number of clones.
    lod : numpy.ndarray or None
        Per-site log-likelihood matrix of shape (M, 3), populated by
        :meth:`lod_scores`.
    lod_germline : numpy.ndarray or None
        Per-site germline LOD scores of shape (M,), populated by
        :meth:`lod_germline`.
    """

    def __init__(self, X):
        assert X.ndim == 3
        if X.shape[2] != 2:
            raise ValueError("Only biallelic variants are tolerated here ...")
        self.X = X
        self.M, self.J, _ = X.shape
        self.p_germline = None
        self.lod = None

    def lod_scores(self, q=30.0):
        """Compute per-site log-likelihoods under three VAF hypotheses.

        Populates ``self.lod`` with shape (M, 3):

        - column 0: log-likelihood under VAF = 0 (no mutation)
        - column 1: log-likelihood under the MLE VAF
        - column 2: log-likelihood under VAF = 0.5 (germline heterozygote)

        Parameters
        ----------
        q : float, optional
            Phred-scaled base quality used to derive the error rate. Must be
            positive. Default is 30.0.
        """
        assert q > 0
        m = self.X.shape[0]
        ll_scores = np.zeros(shape=(m, 3))
        eps = 10 ** (-q / 10.0)
        assert (eps > 0) and (eps < 1)
        for i in range(m):
            alt_reads = self.X[i, :, 1].sum()
            ref_reads = self.X[i, :, 0].sum()
            mle_f = alt_reads / (alt_reads + ref_reads)
            ll_m0 = var_loglik(alt_reads, ref_reads, f=0.0, eps=eps)
            ll_mf = var_loglik(alt_reads, ref_reads, f=mle_f, eps=eps)
            ll_germline = var_loglik(alt_reads, ref_reads, f=0.5, eps=eps)
            ll_scores[i, 0] = ll_m0
            ll_scores[i, 1] = ll_mf
            ll_scores[i, 2] = ll_germline
        self.lod = ll_scores

    def est_germline_prior(self, anno):
        """Set per-site germline priors from a dbSNP-like annotation array.

        Parameters
        ----------
        anno : numpy.ndarray
            Binary or continuous annotation values, shape (M,).

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        assert anno.size == self.M
        self.p_germline = np.ones(self.M) * 5e-5
        # NOTE: we can set t
        raise NotImplementedError("Setting binary priors is not currently implemented")

    def lod_germline(self, p_somatic=3e-6, p_germline=0.095):
        """Compute the per-site LOD score for germline origin.

        LOD_germline = (LL_mle + log p_somatic) - (LL_het + log p_germline),
        converted to base-10 log-odds. A positive LOD score favours somatic
        origin; a negative value favours germline origin.

        Requires :meth:`lod_scores` to have been called first. Result is
        stored in ``self.lod_germline``, shape (M,).

        Parameters
        ----------
        p_somatic : float, optional
            Prior probability of somatic origin. Default is 3e-6.
        p_germline : float, optional
            Prior probability of germline origin. Default is 0.095.
        """
        assert self.lod is not None
        assert (p_somatic > 0) and (p_somatic < 1)
        assert (p_germline > 0) and (p_germline < 1)
        if self.p_germline is None:
            # NOTE: should have some kind of warning here ...
            self.p_germline = np.ones(self.M) * p_germline
        # NOTE: original var_loglik is in base e, so have to convert here ...
        lod_germline = (self.lod[:, 1] + np.log(p_somatic)) - (
            self.lod[:, 2] + np.log(self.p_germline)
        )
        lod_germline = lod_germline / np.log(10)
        self.lod_germline = lod_germline


class BetaOverdispersion:
    """Estimate per-site overdispersion under the Beta-Binomial model.

    Implements the overdispersion test from Spencer-Chapman et al. by fitting
    the rho parameter of a Beta-Binomial distribution to the observed allele
    counts across clones.

    Parameters
    ----------
    X : numpy.ndarray
        Read-count array of shape (M, J, 2). Only biallelic variants are
        supported (``X.shape[2]`` must equal 2).
    """

    def __init__(self, X):
        assert X.ndim == 3
        assert X.shape[2] == 2  # only bi-allelic variants ...
        self.X = X
        self.M, self.J, _ = self.X.shape

    def estimate_rhos(self):
        """Estimate the per-site overdispersion parameter rho.

        For each site, maximises the Beta-Binomial log-likelihood over rho in
        (0, 1) via ``scipy.optimize.minimize_scalar``, where the Beta
        parameters are alpha = phat*(1-rho)/rho and beta = (1-phat)*(1-rho)/rho.

        Returns
        -------
        numpy.ndarray
            Per-site MLE overdispersion values, shape (M,).
        """
        m = self.X.shape[0]
        rhos = np.zeros(m)
        for i in range(m):
            phat = self.X[i, :, 1].sum() / self.X[i, :, :].sum()
            alt_reads = self.X[i, :, 1]
            ref_reads = self.X[i, :, 0]
            opt_rho = minimize_scalar(
                lambda x: (
                    -sum_log_betabinom(
                        alt_reads,
                        ref_reads,
                        phat * (1 - x) / x,
                        (1.0 - phat) * (1 - x) / x,
                    )
                ),
                bounds=(1e-20, 1 - 1e-20),
            ).x
            rhos[i] = opt_rho
        return rhos


class ClonalSim:
    """Simulate clonal sequencing data with germline and somatic variants.

    Generates a synthetic dataset containing a germline sample and a set of
    clonal samples, complete with germline heterozygotes, somatic mutations
    placed on a neutral coalescent genealogy, and realistic read counts.
    Output can be written as a VCF file suitable for use with the CLI.

    Parameters
    ----------
    seq_len : float or int, optional
        Simulated genome length in base-pairs. Default is 1e7.
    n_clones : int, optional
        Number of clonal samples to simulate. Must be > 1. Default is 10.
    """

    def __init__(self, seq_len=1e7, n_clones=10):
        """Initialize the ClonalSim object."""
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
        sd_coverage=5.0,
        mut_rate=1.2e-8,
        q=30,
        seed=42,
    ):
        """Simulate germline heterozygotes and de-novo mutations for the germline sample.

        Draws the number of heterozygous sites from a Poisson distribution,
        samples allele frequencies from a Beta distribution parameterised by
        ``afs``, and simulates read counts under a Normal coverage model.
        Populates ``self.germline_muts``, ``self.germline_af``,
        ``self.germline_alt_reads``, ``self.germline_tot_reads``, and
        ``self.germline_pl``.

        Parameters
        ----------
        afs : list of float, optional
            Shape parameters [a, b] of the Beta prior on allele frequencies in
            the external population. Default is [0.317, 6.067].
        het_rate : float, optional
            Expected heterozygous site density per base-pair. Default is 1e-3.
        mean_coverage : float, optional
            Mean sequencing coverage for the germline sample. Default is 15.0.
        sd_coverage : float, optional
            Standard deviation of sequencing coverage. Default is 5.0.
        mut_rate : float, optional
            De-novo mutation rate per base-pair. Default is 1.2e-8.
        q : int, optional
            Phred-scaled read quality used for genotype likelihood computation.
            Default is 30.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Raises
        ------
        ValueError
            If no heterozygous sites are simulated (increase ``het_rate`` or
            ``seq_len``).
        """
        assert mean_coverage > 0
        assert sd_coverage > 0
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
            norm.rvs(loc=mean_coverage, scale=sd_coverage, size=tot_muts)
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
        """Simulate a somatic genealogy for the clonal samples under a neutral coalescent.

        Uses msprime to simulate a single-locus genealogy for ``n_clones``
        haploid samples. Branch lengths are later rescaled by ``age`` when
        simulating somatic mutations. Populates ``self.genealogy``.

        Parameters
        ----------
        age : int, optional
            Age of the individual at time of sampling (years). Used to rescale
            coalescent branch lengths. Default is 45.
        seed : int, optional
            Random seed forwarded to msprime. Default is 42.
        """
        assert age > 0.0
        assert self.J > 1
        assert seed > 0
        # This simulates a single locus genealogy for clones from a given age
        # NOTE: this is under Ne = 1.0 so we can rescale the branch-lengths accordingly ...
        ts = msprime.sim_ancestry(samples=self.J, ploidy=1, random_seed=42)
        self.genealogy = ts.at(0.0)

    def sim_somatic_mutations(
        self, age=45, mut_rate=5e-9, mean_coverage=15.0, sd_coverage=5.0, q=30, seed=42
    ):
        """Simulate somatic mutations on branches of the clonal genealogy.

        Traverses each branch of ``self.genealogy``, draws the number of
        mutations from a Poisson distribution scaled by the branch length, age,
        genome size, and mutation rate, then assigns read counts to the leaf
        clones that descend from the mutated branch. Populates
        ``self.somatic_muts``, ``self.somatic_alt_reads``,
        ``self.somatic_tot_reads``, and ``self.somatic_mut_pl``.

        Parameters
        ----------
        age : int, optional
            Age of the individual in years, used to rescale branch lengths.
            Default is 45.
        mut_rate : float, optional
            Somatic mutation rate in mutations per base-pair per year (diploid
            rate). Default is 5e-9.
        mean_coverage : float, optional
            Mean sequencing coverage per clone. Default is 15.0.
        sd_coverage : float, optional
            Standard deviation of sequencing coverage. Default is 5.0.
        q : int, optional
            Phred-scaled read quality for genotype likelihood computation.
            Default is 30.
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        """
        assert self.genealogy is not None
        assert age > 0.0
        assert mut_rate > 0.0
        assert seed > 0
        assert mean_coverage > 0.0
        assert sd_coverage > 0.0
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
                    # Sample total read-counts for the somatic mutations ...
                    cur_tot_reads = np.round(
                        norm.rvs(loc=mean_coverage, scale=sd_coverage, size=self.J)
                    ).astype(int)
                    cur_tot_reads[cur_tot_reads <= 0] = 0
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
        self, mean_coverage=15.0, sd_coverage=5.0, q=30, seed=42
    ):
        """Simulate germline heterozygote read counts across all clonal samples.

        For each germline site, draws per-clone coverage from a Normal
        distribution and alt counts from a Binomial(p=0.5) distribution.
        Populates ``self.germline_clone_tot_reads``,
        ``self.germline_clone_alt_reads``, and ``self.germline_clone_pl``.

        Parameters
        ----------
        mean_coverage : float, optional
            Mean sequencing coverage per clone. Default is 15.0.
        sd_coverage : float, optional
            Standard deviation of sequencing coverage. Default is 5.0.
        q : int, optional
            Phred-scaled read quality for genotype likelihood computation.
            Default is 30.
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        """
        assert mean_coverage > 0
        assert sd_coverage > 0
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
                    scale=sd_coverage,
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
        self, mean_coverage=15.0, sd_coverage=5.0, q=30, eps=1e-2, seed=42
    ):
        """Simulate read counts for somatic mutations as seen in the germline sample.

        Draws coverage from a Normal distribution and alt counts from a
        Binomial(p=``eps``) distribution (the germline sample should show only
        error-level alt reads at somatic sites). Populates
        ``self.germline_somatic_tot_reads``, ``self.germline_somatic_alt_reads``,
        and ``self.germline_somatic_pl``.

        Parameters
        ----------
        mean_coverage : float, optional
            Mean sequencing coverage for the germline sample. Default is 15.0.
        sd_coverage : float, optional
            Standard deviation of sequencing coverage. Default is 5.0.
        q : int, optional
            Phred-scaled read quality. Default is 30.
        eps : float, optional
            Error rate used to simulate alt read counts at somatic sites in the
            germline. Default is 1e-2.
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        """
        assert self.n_somatic_mut >= 0
        assert mean_coverage > 0
        assert sd_coverage > 0
        assert q > 0
        assert seed > 0
        assert eps > 0
        np.random.seed(seed)
        somatic_tot_reads = np.round(
            norm.rvs(loc=mean_coverage, scale=sd_coverage, size=self.n_somatic_mut)
        ).astype(int)
        somatic_tot_reads[somatic_tot_reads <= 0] = 0
        somatic_alt_reads = np.zeros(self.n_somatic_mut, dtype=int)
        somatic_pl = np.zeros(shape=(self.n_somatic_mut, 3))
        for i in range(self.n_somatic_mut):
            somatic_alt_reads[i] = binom.rvs(n=somatic_tot_reads[i], p=eps)
            somatic_pl[i, :] = geno_loglik(
                somatic_alt_reads[i], somatic_tot_reads[i], q=q
            )
        self.germline_somatic_tot_reads = somatic_tot_reads
        self.germline_somatic_alt_reads = somatic_alt_reads
        self.germline_somatic_pl = somatic_pl

    def create_read_matrix(self):
        """Build a read-count matrix from the simulated somatic and germline data.

        Stacks somatic sites above germline sites to produce a combined
        array. Each entry stores [ref_reads, alt_reads] per clone.

        Returns
        -------
        numpy.ndarray
            Integer read-count array of shape (M_somatic + M_germline, J, 2),
            where the last dimension is [ref_reads, alt_reads].
        """
        X_somatic = np.array(
            [
                [
                    [
                        self.somatic_tot_reads[i, j] - self.somatic_alt_reads[i, j],
                        self.somatic_alt_reads[i, j],
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.n_somatic_mut)
            ]
        )
        X_germline = np.array(
            [
                [
                    [
                        self.germline_clone_tot_reads[i, j]
                        - self.germline_clone_alt_reads[i, j],
                        self.germline_clone_alt_reads[i, j],
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.n_germline_poly)
            ]
        )
        # Is this the correct stack?
        X = np.vstack([X_somatic, X_germline])
        M, J, L = X.shape
        assert J == self.J
        assert L == 2
        return X

    def create_gt_string(self, alt_reads=0, tot_reads=0, pl=np.array([0, 0, 0])):
        """Format read counts and genotype likelihoods as a VCF genotype field string.

        Determines the GT call from the read counts, formats AD, DP, GQ,
        and PL fields, and returns them as a colon-delimited string.

        Parameters
        ----------
        alt_reads : int, optional
            Number of alternative allele reads. Default is 0.
        tot_reads : int, optional
            Total read depth. Default is 0.
        pl : numpy.ndarray, optional
            Phred-scaled genotype likelihoods [PL(0/0), PL(0/1), PL(1/1)].
            Default is [0, 0, 0].

        Returns
        -------
        gt_str : str
            Full VCF FORMAT field string in GT:AD:DP:GQ:PL format.
        gt : int
            Integer genotype call (0 = hom-ref, 1 = het).
        an : int
            Allele number contribution (0 if missing, 2 otherwise).
        tot_reads : int
            Total read depth.
        gq : float
            Genotype quality (second-lowest minus lowest PL value).
        """
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
        if not np.all(np.isfinite(pl)):
            pl = np.zeros(pl.size)
        ad_str = f"{tot_reads - alt_reads},{alt_reads}"
        dp_str = f"{tot_reads}"
        pl_str = ",".join([str(int(p)) for p in pl])
        gq = np.sort(pl)[1] - np.sort(pl)[0]
        gq_str = f"{int(gq)}"
        return f"{gt_str}:{ad_str}:{dp_str}:{gq_str}:{pl_str}", gt, an, tot_reads, gq

    def write_vcf(self, out=None):
        """Write the simulated variants to a VCF file.

        Produces a VCFv4.2 file containing one germline sample (``Agermline``)
        followed by J clonal samples (``Aclone0`` … ``AcloneN``). Germline
        heterozygotes are written first, then somatic variants. The INFO
        field includes AC, AF, AN, DP, ExternalAF, and SM (somatic indicator).

        Parameters
        ----------
        out : str
            Output file path. Must be writable.
        """
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
##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth some reads may have been filtered">
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
