"""Tests that the EM objective is a valid minorant and the ascent holds.

The carrier indicator ``c_kj`` exists only when the site is not a germline
het, and the E-step returns ``gamma_kj`` conditional on that branch, so the
expected complete-data log-likelihood weights every carrier term by
``(1 - eta_k)``.  Dropping that factor leaves an objective that is not a
minorant of the observed log-likelihood, and EM stops being monotone.
"""

import numpy as np
import pytest

from poly_utils import kappa_Q, kappa_score
from pGermlinePoly.pGermlinePoly import ProbGermline


def _simulate_mixture(n_sites=600, n_clones=10, depth=14, frac_het=0.4, seed=0):
    """Simulate a germline/non-germline mixture with matching annotations."""
    rng = np.random.default_rng(seed)
    is_het = rng.random(n_sites) < frac_het
    X = np.empty((n_sites, n_clones, 2), dtype=np.int64)
    for k in range(n_sites):
        if is_het[k]:
            alt = rng.binomial(depth, 0.5, n_clones)
        else:
            carrier = rng.random(n_clones) < 0.25
            alt = np.where(
                carrier,
                rng.binomial(depth, 0.5, n_clones),
                rng.binomial(depth, 1e-3, n_clones),
            )
        X[k, :, 0] = depth - alt
        X[k, :, 1] = alt
    # An informative annotation, plus intercept.
    anno = is_het * 1.0 + rng.normal(0, 0.5, n_sites)
    Theta = np.column_stack([np.ones(n_sites), anno])
    return X, Theta


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_em_observed_loglikelihood_is_nondecreasing(seed):
    """The observed log-likelihood must not decrease across EM iterations."""
    X, Theta = _simulate_mixture(seed=seed)
    obj = ProbGermline(X=X, Theta=Theta)
    loglls, _, _, _ = obj.em_algo(lambdas=np.zeros(2), max_iter=25)
    deltas = np.diff(loglls)
    # Allow only floating-point noise relative to the objective's magnitude.
    tol = 1e-9 * np.abs(loglls).max()
    assert deltas.min() >= -tol, (
        f"EM decreased by {deltas.min():.3e} (tolerance {tol:.3e}); "
        f"the M-step objective is not minorising the observed likelihood."
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_em_monotone_with_betabinomial_het_component(seed):
    """Monotonicity also holds once the het component is a Beta-Binomial."""
    X, Theta = _simulate_mixture(seed=seed)
    obj = ProbGermline(X=X, Theta=Theta, het_conc=30.0, het_mode="clone")
    loglls, _, _, _ = obj.em_algo(lambdas=np.zeros(2), max_iter=25)
    deltas = np.diff(loglls)
    tol = 1e-9 * np.abs(loglls).max()
    assert deltas.min() >= -tol


def test_m_step_increases_the_q_objective():
    """One M-step must not decrease Q evaluated at fixed responsibilities."""
    X, Theta = _simulate_mixture(seed=3)
    obj = ProbGermline(X=X, Theta=Theta)
    lam = np.array([-1.0, 0.3])
    eta, gammas = obj._e_step(lam, np.zeros(0), obj.kappa)

    def q_value(lambdas):
        logit_pi = Theta @ lambdas
        logit_phi = np.broadcast_to(logit_pi[:, None], (obj.M, obj.J))
        log_pi = -np.log1p(np.exp(-logit_pi))
        log1m_pi = -np.log1p(np.exp(logit_pi))
        log_phi = -np.log1p(np.exp(-logit_phi))
        log1m_phi = -np.log1p(np.exp(logit_phi))
        site = np.dot(eta, log_pi) + np.dot(1.0 - eta, log1m_pi)
        clone = (
            (1.0 - eta)[:, None]
            * (gammas * log_phi + (1.0 - gammas) * log1m_phi)
        ).sum()
        return site + clone

    lam_new, _ = obj.m_step_lambda_beta(eta, gammas, lam, np.zeros(0))
    assert q_value(lam_new) >= q_value(lam) - 1e-8


@pytest.mark.parametrize("fn", [kappa_Q, kappa_score])
def test_kappa_objective_eta_defaults_to_unweighted(fn):
    """Omitting eta is equivalent to eta = 0, preserving the old behaviour."""
    rng = np.random.default_rng(4)
    X = rng.integers(0, 12, (50, 6, 2)).astype(np.int64)
    gammas = rng.random((50, 6))
    assert fn(X, gammas, 1e-3, 10.0) == fn(X, gammas, 1e-3, 10.0, np.zeros(50))


@pytest.mark.parametrize("fn", [kappa_Q, kappa_score])
def test_kappa_objective_eta_weighting_is_exact(fn):
    """Weighting by (1 - eta_k) equals folding it into (1 - gamma_kj)."""
    rng = np.random.default_rng(5)
    X = rng.integers(0, 12, (50, 6, 2)).astype(np.int64)
    gammas = rng.random((50, 6))
    eta = rng.random(50)
    folded = np.ascontiguousarray(1.0 - (1.0 - eta)[:, None] * (1.0 - gammas))
    assert fn(X, gammas, 1e-3, 10.0, eta) == pytest.approx(
        fn(X, folded, 1e-3, 10.0)
    )


def test_kappa_score_ignores_sites_with_eta_one():
    """A site certain to be germline contributes nothing to the kappa score."""
    rng = np.random.default_rng(6)
    X = rng.integers(1, 12, (30, 5, 2)).astype(np.int64)
    gammas = rng.random((30, 5))
    eta = np.ones(30)
    assert kappa_score(X, gammas, 1e-3, 10.0, eta) == pytest.approx(0.0)
