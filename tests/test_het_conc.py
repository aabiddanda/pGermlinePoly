"""Tests for the Beta-Binomial heterozygote component and its calibration."""

import numpy as np
import pytest
from scipy.special import gammaln

from poly_utils import logprob_het, logprob_het_bb, het_conc_Q, het_conc_score
from pGermlinePoly.pGermlinePoly import ProbGermline


@pytest.fixture
def counts():
    """Return alt/ref count vectors for a single site across 12 clones."""
    rng = np.random.default_rng(7)
    ax = rng.integers(0, 20, 12).astype(np.int64)
    rx = rng.integers(0, 20, 12).astype(np.int64)
    return ax, rx


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("conc", [np.inf, 0.0, -1.0, 1e13])
def test_point_mass_sentinels_recover_logprob_het(counts, mode, conc):
    """Non-finite, non-positive, and huge concentrations reduce to p = 0.5."""
    ax, rx = counts
    assert logprob_het_bb(ax, rx, conc, mode) == pytest.approx(logprob_het(ax, rx))


@pytest.mark.parametrize("conc", [0.5, 5.0, 50.0, 500.0])
def test_pooled_kernel_matches_closed_form(counts, conc):
    """Site mode equals a Beta-Binomial on the pooled counts."""
    ax, rx = counts
    h, A, N = conc / 2.0, ax.sum(), (ax + rx).sum()
    expected = (
        gammaln(A + h)
        + gammaln(N - A + h)
        - gammaln(N + conc)
        + gammaln(conc)
        - 2 * gammaln(h)
    )
    assert logprob_het_bb(ax, rx, conc, 0) == pytest.approx(expected)


@pytest.mark.parametrize("conc", [0.5, 5.0, 50.0, 500.0])
def test_per_clone_kernel_matches_closed_form(counts, conc):
    """Clone mode equals a product of per-clone Beta-Binomials."""
    ax, rx = counts
    h = conc / 2.0
    expected = (
        gammaln(ax + h)
        + gammaln(rx + h)
        - gammaln(ax + rx + conc)
        + gammaln(conc)
        - 2 * gammaln(h)
    ).sum()
    assert logprob_het_bb(ax, rx, conc, 1) == pytest.approx(expected)


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("conc", [3.0, 30.0, 300.0])
def test_het_conc_score_matches_numerical_gradient(mode, conc):
    """The analytic score agrees with a central difference of the objective."""
    rng = np.random.default_rng(11)
    X = rng.integers(0, 14, (40, 6, 2)).astype(np.int64)
    eta = rng.random(40)
    eps = conc * 1e-6
    numeric = (
        het_conc_Q(X, eta, conc + eps, mode) - het_conc_Q(X, eta, conc - eps, mode)
    ) / (2 * eps)
    assert het_conc_score(X, eta, conc, mode) == pytest.approx(numeric, rel=1e-4)


@pytest.mark.parametrize("mode", [0, 1])
def test_het_conc_score_is_zero_at_point_mass(mode):
    """The objective does not depend on c once the het component is a point mass."""
    rng = np.random.default_rng(3)
    X = rng.integers(0, 14, (20, 5, 2)).astype(np.int64)
    eta = rng.random(20)
    assert het_conc_score(X, eta, np.inf, mode) == 0.0


def _simulate_hets(n_sites, n_clones, depth, conc, seed=0):
    """Simulate germline hets whose per-clone VAF is Beta(c/2, c/2)."""
    rng = np.random.default_rng(seed)
    p = rng.beta(conc / 2.0, conc / 2.0, size=(n_sites, n_clones))
    alt = rng.binomial(depth, p)
    X = np.empty((n_sites, n_clones, 2), dtype=np.int64)
    X[:, :, 0] = depth - alt
    X[:, :, 1] = alt
    return X


@pytest.mark.parametrize("true_conc", [10.0, 25.0, 60.0])
def test_calibration_recovers_simulated_concentration(true_conc):
    """Label-free calibration recovers the generating concentration."""
    X = _simulate_hets(4000, 30, 14, true_conc, seed=int(true_conc))
    Theta = np.ones((X.shape[0], 1))
    obj = ProbGermline(X=X, Theta=Theta, het_conc=np.inf, het_mode="clone")
    # No VAF window: the simulated set is pure, so selection would only bias it.
    est = obj.calibrate_het_conc(
        method="window", vaf_window=(0.0, 1.0), min_records=100
    )
    assert est == pytest.approx(true_conc, rel=0.15)


def test_window_selection_is_flip_invariant():
    """The symmetric VAF window is unchanged by minor-allele reorientation."""
    X = _simulate_hets(500, 20, 12, 30.0, seed=5)
    Theta = np.ones((X.shape[0], 1))
    obj = ProbGermline(X=X, Theta=Theta, het_mode="clone")
    before = obj._het_calibration_mask("window")
    obj.reorient_to_minor_allele()
    assert np.array_equal(before, obj._het_calibration_mask("window"))


def test_em_does_not_touch_het_conc_by_default():
    """``fit_het_conc`` defaults to False, so EM leaves the concentration alone."""
    rng = np.random.default_rng(2)
    X = rng.integers(0, 12, (300, 8, 2)).astype(np.int64)
    Theta = np.column_stack([np.ones(300), rng.random(300)])
    obj = ProbGermline(X=X, Theta=Theta, het_conc=250.0, het_mode="clone")
    obj.em_algo(lambdas=np.zeros(2), max_iter=3)
    assert obj.het_conc == 250.0


def test_invalid_het_mode_rejected():
    """Only 'site' and 'clone' are accepted."""
    X = np.ones((5, 3, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="het_mode"):
        ProbGermline(X=X, Theta=np.ones((5, 1)), het_mode="pooled")


def test_invalid_calibration_method_rejected():
    """An unknown selection method raises rather than silently defaulting."""
    X = np.ones((5, 3, 2), dtype=np.int64)
    obj = ProbGermline(X=X, Theta=np.ones((5, 1)))
    with pytest.raises(ValueError, match="method"):
        obj._het_calibration_mask(method="bogus")
