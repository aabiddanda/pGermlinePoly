"""Tests for the ReadCountUtils mixin.

ReadCountUtils is not exported at the package level; tests use a minimal
concrete subclass (_RC) to exercise the mixin directly without the overhead
of ProbGermline's annotation machinery or BetaOverdispersion's optimiser.
"""

import numpy as np
import pytest

from pGermlinePoly.pGermlinePoly import ReadCountUtils


# ---------------------------------------------------------------------------
# Minimal concrete subclass for direct mixin testing
# ---------------------------------------------------------------------------


class _RC(ReadCountUtils):
    """Thin wrapper that exposes ReadCountUtils through the smallest possible init."""

    def __init__(self, X):
        self.X = self.validate_X(X)
        self.M, self.J, _ = self.X.shape


def _make_X(ref, alt, n_clones=4):
    """Build a (1, n_clones, 2) array with uniform ref/alt counts."""
    X = np.zeros((1, n_clones, 2), dtype=np.int64)
    X[:, :, 0] = ref
    X[:, :, 1] = alt
    return X


# ---------------------------------------------------------------------------
# validate_X tests
# ---------------------------------------------------------------------------


def testvalidate_X_valid_returns_int64_contiguous():
    X = np.ones((5, 3, 2), dtype=np.float32)
    out = ReadCountUtils.validate_X(X)
    assert out.dtype == np.int64
    assert out.flags["C_CONTIGUOUS"]
    assert out.shape == (5, 3, 2)


def testvalidate_X_preserves_values():
    X = np.array([[[10, 20], [30, 40]]], dtype=np.int32)
    out = ReadCountUtils.validate_X(X)
    np.testing.assert_array_equal(out, X)


def testvalidate_X_wrong_ndim_raises():
    with pytest.raises(ValueError, match="shape"):
        ReadCountUtils.validate_X(np.ones((5, 3)))


def testvalidate_X_wrong_last_dim_raises():
    with pytest.raises(ValueError, match="shape"):
        ReadCountUtils.validate_X(np.ones((5, 3, 3)))


def testvalidate_X_1d_raises():
    with pytest.raises(ValueError, match="shape"):
        ReadCountUtils.validate_X(np.ones(10))


def testvalidate_X_already_int64_still_contiguous():
    """Even when the dtype is already correct the result must be contiguous."""
    base = np.ones((4, 2, 2), dtype=np.int64)
    sliced = base[:, :, :]  # still contiguous, but a view
    out = ReadCountUtils.validate_X(sliced)
    assert out.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# pooled_vaf tests
# ---------------------------------------------------------------------------


def test_pooled_vaf_shape():
    rc = _RC(np.zeros((10, 5, 2), dtype=np.int64))
    assert rc.pooled_vaf.shape == (10,)


def test_pooled_vaf_known_values():
    """alt=30, ref=70 across 3 clones → pooled VAF = 30/100 = 0.3."""
    X = np.array([[[70, 30], [70, 30], [70, 30]]], dtype=np.int64)
    rc = _RC(X)
    np.testing.assert_allclose(rc.pooled_vaf, [0.3])


def test_pooled_vaf_all_alt():
    X = np.array([[[0, 50], [0, 50]]], dtype=np.int64)
    rc = _RC(X)
    np.testing.assert_allclose(rc.pooled_vaf, [1.0])


def test_pooled_vaf_all_ref():
    X = np.array([[[50, 0], [50, 0]]], dtype=np.int64)
    rc = _RC(X)
    np.testing.assert_allclose(rc.pooled_vaf, [0.0])


def test_pooled_vaf_zero_depth_returns_zero():
    """Sites with zero total depth are clipped to 1 in the denominator → VAF = 0."""
    X = np.zeros((3, 4, 2), dtype=np.int64)
    rc = _RC(X)
    np.testing.assert_allclose(rc.pooled_vaf, [0.0, 0.0, 0.0])


def test_pooled_vaf_mixed_sites():
    X = np.array(
        [
            [[70, 30], [70, 30]],  # 30 %
            [[0, 100], [0, 100]],  # 100 %
            [[100, 0], [100, 0]],  # 0 %
        ],
        dtype=np.int64,
    )
    rc = _RC(X)
    np.testing.assert_allclose(rc.pooled_vaf, [0.3, 1.0, 0.0], atol=1e-9)


def test_pooled_vaf_is_float64():
    rc = _RC(np.ones((2, 2, 2), dtype=np.int64))
    assert rc.pooled_vaf.dtype == np.float64


# ---------------------------------------------------------------------------
# reorient_to_minor_allele tests
# ---------------------------------------------------------------------------


def test_reorient_flipped_mask_boundary():
    """Exactly 50 % alt is NOT flipped (strict > 0.5 threshold)."""
    X = np.array(
        [
            [[50, 50], [50, 50]],  # 50 % — not flipped
            [[30, 70], [30, 70]],  # 70 % — flipped
            [[80, 20], [80, 20]],  # 20 % — not flipped
        ],
        dtype=np.int64,
    )
    rc = _RC(X)
    rc.reorient_to_minor_allele()

    assert not rc.flipped[0], "50 % should not be flipped"
    assert rc.flipped[1], "70 % should be flipped"
    assert not rc.flipped[2], "20 % should not be flipped"


def test_reorient_columns_swapped_correctly():
    """After a flip the new alt column equals the original ref column."""
    ref, alt = 30, 70
    X = _make_X(ref=ref, alt=alt, n_clones=5)
    rc = _RC(X)
    rc.reorient_to_minor_allele()

    assert rc.flipped[0]
    np.testing.assert_array_equal(rc.X[0, :, 1], ref)  # new alt = old ref
    np.testing.assert_array_equal(rc.X[0, :, 0], alt)  # new ref = old alt


def test_reorient_pooled_vaf_le_half_after():
    """After re-orientation every site has pooled VAF ≤ 0.5."""
    rng = np.random.default_rng(42)
    M, J = 50, 8
    n = rng.integers(10, 60, size=(M, J))
    a = rng.integers(0, n + 1)
    X = np.stack([n - a, a], axis=-1).astype(np.int64)
    rc = _RC(X)
    rc.reorient_to_minor_allele()
    assert np.all(rc.pooled_vaf <= 0.5 + 1e-9)


def test_reorient_flipped_attr_absent_before_call():
    rc = _RC(_make_X(ref=80, alt=20))
    assert not hasattr(rc, "flipped")
    rc.reorient_to_minor_allele()
    assert hasattr(rc, "flipped")
    assert rc.flipped.shape == (1,)
    assert rc.flipped.dtype == bool


def test_reorient_no_sites_to_flip():
    X = _make_X(ref=80, alt=20, n_clones=6)
    rc = _RC(X)
    rc.reorient_to_minor_allele()
    assert not rc.flipped.any()
    np.testing.assert_array_equal(rc.X[0, :, 1], 20)  # unchanged


def test_reorient_all_sites_flipped():
    X = np.stack([_make_X(ref=20, alt=80, n_clones=4)[0]] * 5)
    rc = _RC(X)
    rc.reorient_to_minor_allele()
    assert rc.flipped.all()
    np.testing.assert_array_equal(rc.X[:, :, 1], 20)


def test_reorient_symmetry():
    """Mirror-image inputs produce identical pooled_vaf after re-orientation."""
    rng = np.random.default_rng(7)
    M, J = 20, 6
    n = rng.poisson(40, size=(M, J))
    a = rng.binomial(n, 0.3)
    X_low = np.stack([n - a, a], axis=-1).astype(np.int64)
    X_high = np.stack([a, n - a], axis=-1).astype(np.int64)

    rc_low = _RC(X_low)
    rc_low.reorient_to_minor_allele()

    rc_high = _RC(X_high)
    rc_high.reorient_to_minor_allele()

    np.testing.assert_allclose(rc_low.pooled_vaf, rc_high.pooled_vaf, atol=1e-10)
    np.testing.assert_array_equal(rc_low.X, rc_high.X)


# ---------------------------------------------------------------------------
# Inheritance sanity checks — confirm all three model classes use ReadCountUtils
# ---------------------------------------------------------------------------


def test_prob_germline_inherits_read_count_utils():
    from pGermlinePoly import ProbGermline

    assert issubclass(ProbGermline, ReadCountUtils)


def test_beta_overdispersion_inherits_read_count_utils():
    from pGermlinePoly import BetaOverdispersion

    assert issubclass(BetaOverdispersion, ReadCountUtils)


def test_mutect_lod_inherits_read_count_utils():
    from pGermlinePoly import MutectLOD

    assert issubclass(MutectLOD, ReadCountUtils)


def test_read_count_utils_not_in_top_level_namespace():
    import pGermlinePoly

    assert not hasattr(pGermlinePoly, "ReadCountUtils"), (
        "ReadCountUtils should not be exported at the top-level package namespace"
    )


def testvalidate_X_accessible_on_all_model_classes():
    """validate_X is callable on each concrete class without instantiation."""
    from pGermlinePoly import ProbGermline, BetaOverdispersion, MutectLOD

    X = np.ones((3, 2, 2), dtype=np.float32)
    for cls in (ProbGermline, BetaOverdispersion, MutectLOD):
        out = cls.validate_X(X)
        assert out.dtype == np.int64


def test_pooled_vaf_accessible_on_all_model_classes():
    from pGermlinePoly import BetaOverdispersion, MutectLOD

    X = np.ones((4, 3, 2), dtype=np.int64) * 10
    for cls in (BetaOverdispersion, MutectLOD):
        obj = cls(X=X)
        vaf = obj.pooled_vaf
        assert vaf.shape == (4,)
        np.testing.assert_allclose(vaf, 0.5)
