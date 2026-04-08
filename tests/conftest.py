import numpy as np


def sim_read_counts(m=100, j=20, coverage=30, p_somatic=0.0, seed=42):
    """Simulate matrices of read-counts for use as inputs."""
    assert seed > 0
    assert m > 0
    assert j > 1
    assert coverage > 0
    np.random.seed(seed)
    X = np.zeros((m, j, 2), dtype="int")
    for i in range(m):
        n_i = np.random.poisson(coverage, size=j)
        a_i = np.random.binomial(n=n_i, p=0.5)
        X[i, :, 1] = a_i
        X[i, :, 0] = n_i - a_i
    return X


def sim_annotations(m=100, a=5, seed=42):
    """Simulate matrices of annotations as normal variables."""
    assert seed > 0
    assert m > 0
    assert a > 0
    np.random.seed(seed)
    A = np.random.normal(loc=np.random.normal(size=a), size=(m, a))
    return A
