import numpy as np
import pytest

from pGermlinePoly import BetaOverdispersion, ClonalSim

# The second variant should have a high-overdispersion
X = np.array(
    [
        [[5, 5], [5, 5], [5, 5]],
        [[5, 0], [5, 0], [5, 5]],
    ],
    dtype=np.uint8,
)


@pytest.mark.parametrize("m,j", [(10, 4)])
def test_initialization(m, j):
    """Test that the intialization of the class will go well."""
    X = np.zeros(shape=(m, j, 2), dtype=np.uint8)
    betaoverdisp = BetaOverdispersion(X=X)
    assert betaoverdisp.J == j
    assert betaoverdisp.M == m


@pytest.mark.parametrize("X", [X])
def test_rhos(X):
    """Test that the intialization of the class will go well."""
    betaoverdisp = BetaOverdispersion(X=X)
    rhos = betaoverdisp.estimate_rhos()
    assert rhos.ndim == 1
    assert rhos.size == X.shape[0]
    if rhos.size == 2:
        assert rhos[1] >= rhos[0]


def test_betabinom_from_sim():
    clone_sim = ClonalSim(seq_len=1e6, n_clones=10)
    clone_sim.simulate_germline(seed=42)
    clone_sim.simulate_clone_genealogy(age=80, seed=42)
    clone_sim.sim_somatic_mutations(age=80, mut_rate=1e-6, seed=42)
    assert clone_sim.n_somatic_mut > 0
    clone_sim.simulate_germline_somatic_muts()
    clone_sim.simulate_clonal_germline_muts()
    # Can we make a read-matrix from the clonal sims as well?
    X = clone_sim.create_read_matrix()
