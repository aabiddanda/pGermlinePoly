import pytest

from pGermlinePoly import ClonalSim


@pytest.mark.parametrize("seqlen,n", [(0.0, 0), (1e6, 0), (1e6, -0.5), (-100, 3)])
def test_clonal_sim_bad_params(seqlen, n):
    with pytest.raises(Exception):
        ClonalSim(seq_len=seqlen, n_clones=n)


@pytest.mark.parametrize("seqlen,n", [(1e6, 2), (1e6, 1000), (1e8, 50)])
def test_clonal_sim(seqlen, n):
    clone_sim = ClonalSim(seq_len=seqlen, n_clones=n)
    assert clone_sim.seq_len == seqlen
    assert clone_sim.J == n
