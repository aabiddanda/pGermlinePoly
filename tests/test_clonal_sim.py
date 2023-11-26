import pytest

from pGermlinePoly import ClonalSim


@pytest.mark.parametrize("seqlen,n", [(0.0, 0), (1e6, 0), (1e6, -0.5), (-100, 3)])
def test_clonal_sim_bad_params(seqlen, n):
    with pytest.raises(Exception):
        ClonalSim(seq_len=seqlen, n_clones=n)
