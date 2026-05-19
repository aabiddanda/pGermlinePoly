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
