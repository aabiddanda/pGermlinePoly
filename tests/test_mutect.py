import numpy as np
from pGermlinePoly import MutectLOD
import pytest

X = np.array(
    [
        [[2, 3], [3, 4]],
        [[1, 0], [8, 8]],
        [[3, 3], [5, 5]],
    ],
    dtype="int",
)
# TODO: write a function for simulating read-count matrices


@pytest.mark.parametrize("X", [X])
def test_init_mutect(X):
    """Test initialization ."""
    mutect = MutectLOD(X=X)
    assert mutect.M == X.shape[0]


@pytest.mark.parametrize("X", [X])
@pytest.mark.parametrize("q", [10, 20, 30])
def test_lod_calc(X, q):
    """Test the calculation of"""
    mutect = MutectLOD(X=X)
    assert mutect.M == X.shape[0]
    mutect.lod_scores(q=q)
    assert mutect.lod is not None
    assert mutect.lod.ndim == 2
    assert mutect.lod.shape[0] == X.shape[0]


@pytest.mark.parametrize("X", [X])
@pytest.mark.parametrize("q", [10, 20, 30])
def test_ll_germline(X, q):
    mutect = MutectLOD(X=X)
    mutect.lod_scores(q=q)
    mutect.lod_germline()
    assert mutect.lod_germline.size == X.shape[0]
    assert np.all(~np.isnan(mutect.lod_germline))
