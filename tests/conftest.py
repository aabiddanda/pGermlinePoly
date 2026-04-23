import numpy as np


def sim_read_counts(
    m=100, j=20, coverage=30, p_somatic=0.0, vaf=0.1, eps=1e-3, seed=42
):
    """Simulate matrices of read-counts for use as inputs."""
    assert seed > 0
    assert m > 0
    assert j > 1
    assert coverage > 0
    assert (eps >= 0) and (eps < 0.5)
    assert (vaf > 0.0) and (vaf < 1.0)
    np.random.seed(seed)
    X = np.zeros((m, j, 2), dtype="int")
    somatic = np.random.binomial(n=1, p=p_somatic, size=m)
    Z = np.zeros((m, j), dtype="int")
    for i in range(m):
        n_i = np.random.poisson(coverage, size=j)
        if somatic[i]:
            x_i = np.random.binomial(n=1, p=vaf, size=j)
            Z[i, :] = x_i
            a_i = np.zeros(j)
            for jx, x in enumerate(x_i):
                if x:
                    a_i[jx] = np.random.binomial(n=n_i[jx], p=0.5)
                else:
                    a_i[jx] = np.random.binomial(n=n_i[jx], p=eps)
        else:
            a_i = np.random.binomial(n=n_i, p=0.5)
        X[i, :, 1] = a_i
        X[i, :, 0] = n_i - a_i
    return X, somatic, Z


def sim_annotations(m=100, a=5, seed=42):
    """Simulate matrices of annotations as normal variables."""
    assert seed > 0
    assert m > 0
    assert a > 0
    np.random.seed(seed)
    A = np.random.normal(loc=np.random.normal(size=a), size=(m, a))
    return A
