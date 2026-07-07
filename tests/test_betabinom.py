import numpy as np
import pytest
from conftest import sim_read_counts
from pGermlinePoly import BetaOverdispersion, ClonalSim


@pytest.mark.parametrize("m,j", [(10, 4)])
def test_initialization(m, j):
    """Test initialization of the class."""
    X = np.zeros(shape=(m, j, 2), dtype=np.uint8)
    betaoverdisp = BetaOverdispersion(X=X)
    assert betaoverdisp.J == j
    assert betaoverdisp.M == m


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
def test_rhos_germline(m, j, c):
    """Test estimation of rhos."""
    X, *_ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    betaoverdisp = BetaOverdispersion(X=X)
    rhos = betaoverdisp.estimate_rhos()
    assert rhos.ndim == 1
    assert rhos.size == X.shape[0]
    assert ~np.all(rhos == rhos[0])


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.25])
@pytest.mark.parametrize("v", [0.05, 0.1, 0.25])
def test_rhos_somatic(m, j, c, p, v):
    """Test estimation of rhos."""
    X, somatic, _ = sim_read_counts(
        m=m, j=j, coverage=c, p_somatic=p, vaf=v, seed=m + j
    )
    betaoverdisp = BetaOverdispersion(X=X)
    rhos = betaoverdisp.estimate_rhos()
    assert rhos.ndim == 1
    assert rhos.size == X.shape[0]
    assert ~np.all(rhos == rhos[0])
    if np.sum(somatic) > 0:
        assert np.mean(rhos[somatic == 1]) > 0.1


def test_betabinom_from_sim():
    clone_sim = ClonalSim(seq_len=1e6, n_clones=10)
    clone_sim.simulate_germline(seed=42)
    clone_sim.simulate_clone_genealogy(age=80, seed=42)
    clone_sim.sim_somatic_mutations(age=80, mut_rate=1e-6, seed=42)
    assert clone_sim.n_somatic_mut > 0
    clone_sim.simulate_germline_somatic_muts()
    clone_sim.simulate_clonal_germline_muts()
    X = clone_sim.create_read_matrix()
    betaoverdisp = BetaOverdispersion(X=X)
    rhos = betaoverdisp.estimate_rhos()
    assert rhos.ndim == 1
    assert rhos.size == X.shape[0]
    assert ~np.all(rhos == rhos[0])


# ---------------------------------------------------------------------------
# reorient_to_minor_allele tests
# ---------------------------------------------------------------------------


def test_reorient_flipped_mask():
    """Sites with pooled alt > 50 % are flagged; others are not."""
    X = np.array(
        [
            [[20, 80], [20, 80]],  # alt=80 % → flipped
            [[80, 20], [80, 20]],  # alt=20 % → not flipped
            [[50, 50], [50, 50]],  # alt=50 % → not flipped (strict >)
        ],
        dtype=np.int64,
    )
    bd = BetaOverdispersion(X=X)
    bd.reorient_to_minor_allele()

    assert bd.flipped[0]
    assert not bd.flipped[1]
    assert not bd.flipped[2]


def test_reorient_minor_allele_always_le_half():
    """After re-orientation every site has pooled alt frequency ≤ 0.5."""
    rng = np.random.default_rng(1)
    M, J = 30, 6
    n = rng.integers(10, 50, size=(M, J))
    a = rng.integers(0, n + 1)
    X = np.stack([n - a, a], axis=-1).astype(np.int64)
    bd = BetaOverdispersion(X=X)
    bd.reorient_to_minor_allele()

    pooled_alt = bd.X[:, :, 1].sum(axis=1)
    pooled_tot = bd.X.sum(axis=(1, 2))
    assert np.all(pooled_alt / pooled_tot <= 0.5 + 1e-9)


def test_reorient_rho_symmetric():
    """rho should be the same whether alt=p or alt=1-p (BB is symmetric in alt/ref).

    This validates that re-orientation does not distort the overdispersion estimate.
    """
    rng = np.random.default_rng(3)
    M, J, cov = 20, 8, 40
    n = rng.poisson(cov, size=(M, J))
    a_low = rng.binomial(n, 0.3)
    X_low = np.stack([n - a_low, a_low], axis=-1).astype(np.int64)
    X_high = np.stack([a_low, n - a_low], axis=-1).astype(np.int64)

    bd_low = BetaOverdispersion(X=X_low)
    bd_low.reorient_to_minor_allele()
    rhos_low = bd_low.estimate_rhos()

    bd_high = BetaOverdispersion(X=X_high)
    bd_high.reorient_to_minor_allele()
    rhos_high = bd_high.estimate_rhos()

    assert np.allclose(rhos_low, rhos_high, atol=1e-6), (
        "rho must be the same for mirror-image datasets after re-orientation"
    )
