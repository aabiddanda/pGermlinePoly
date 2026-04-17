from pGermlinePoly import MutectLOD
from conftest import sim_read_counts
import numpy as np
import pytest


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
def test_init_mutect(m, j, c):
    """Test the initalization."""
    X, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    mutect = MutectLOD(X=X)
    assert mutect.M == X.shape[0]


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("q", [10, 20, 30])
def test_lod_calc(m, j, c, q):
    """Test the calculation of the LOD components."""
    X, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    mutect = MutectLOD(X=X)
    assert mutect.M == X.shape[0]
    mutect.lod_scores(q=q)
    assert mutect.lod is not None
    assert mutect.lod.ndim == 2
    assert mutect.lod.shape[0] == X.shape[0]


@pytest.mark.parametrize("m", [10, 50, 100])
@pytest.mark.parametrize("j", [5, 50, 100])
@pytest.mark.parametrize("c", [5, 10, 30])
@pytest.mark.parametrize("q", [10, 20, 30])
def test_ll_germline(m, j, c, q):
    X, _ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
    mutect = MutectLOD(X=X)
    mutect.lod_scores(q=q)
    mutect.lod_germline()
    assert mutect.lod_germline.size == X.shape[0]
    assert np.all(~np.isnan(mutect.lod_germline))


# @pytest.mark.parametrize("m", [10, 50, 100])
# @pytest.mark.parametrize("j", [5, 50, 100])
# @pytest.mark.parametrize("c", [5, 10, 30])
# @pytest.mark.parametrize("q", [10, 20, 30])
# def test_ll_germline(m, j, c, q):
#     X,_ = sim_read_counts(m=m, j=j, coverage=c, seed=m + j)
#     mutect = MutectLOD(X=X)
#     mutect.lod_scores(q=q)
#     mutect.lod_germline()
#     assert mutect.lod_germline.size == X.shape[0]
#     assert np.all(~np.isnan(mutect.lod_germline))
