import numpy as np
import pytest
from cyvcf2 import VCF

from pGermlinePoly import ClonalSim
from poly_utils import geno_loglik, geno_loglik_2d


@pytest.mark.parametrize("seqlen,n", [(0.0, 0), (1e6, 0), (1e6, -0.5), (-100, 3)])
def test_clonal_sim_bad_params(seqlen, n):
    with pytest.raises(Exception):
        ClonalSim(seq_len=seqlen, n_clones=n)


@pytest.mark.parametrize("seqlen,n", [(1e6, 2), (1e6, 1000), (1e8, 50)])
def test_clonal_sim(seqlen, n):
    clone_sim = ClonalSim(seq_len=seqlen, n_clones=n)
    assert clone_sim.seq_len == seqlen
    assert clone_sim.J == n


@pytest.mark.parametrize("seqlen,n", [(1e6, 2), (1e6, 1000), (1e6, 50), (1e6, 500)])
def test_genealogy(seqlen, n):
    clone_sim = ClonalSim(seq_len=seqlen, n_clones=n)
    assert clone_sim.seq_len == seqlen
    assert clone_sim.J == n
    clone_sim.simulate_clone_genealogy()
    assert clone_sim.genealogy.num_samples(clone_sim.genealogy.root) == clone_sim.J


@pytest.mark.parametrize("seqlen,n", [(1e6, 2), (1e6, 50), (1e6, 500)])
def test_germline_mutgen(seqlen, n):
    """Test simulation of germline mutations."""
    clone_sim = ClonalSim(seq_len=seqlen, n_clones=n)
    clone_sim.simulate_germline()
    assert clone_sim.n_germline_poly > 0


@pytest.mark.parametrize("seqlen,n", [(10e6, 2), (10e6, 50), (10e6, 500)])
def test_somatic_mutgen(seqlen, n):
    """Test simulation of germline mutations."""
    clone_sim = ClonalSim(seq_len=seqlen, n_clones=n)
    clone_sim.simulate_clone_genealogy()
    clone_sim.sim_somatic_mutations(mut_rate=1e-6)
    assert clone_sim.n_somatic_mut > 0


@pytest.mark.parametrize("seqlen,n", [(10e6, 2), (10e6, 50), (10e6, 500)])
def test_somatic_germline_mutgen(seqlen, n):
    """Test simulation of germline status at the somatic mutations."""
    clone_sim = ClonalSim(seq_len=seqlen, n_clones=n)
    clone_sim.simulate_clone_genealogy()
    clone_sim.sim_somatic_mutations(mut_rate=1e-6)
    assert clone_sim.n_somatic_mut > 0
    clone_sim.simulate_germline_somatic_muts()
    assert np.any(clone_sim.germline_somatic_pl != 0)


@pytest.mark.parametrize("seqlen,n", [(1e6, 10)])
def test_full_germline_somatic_sim(seqlen, n):
    clone_sim = ClonalSim(seq_len=seqlen, n_clones=n)
    clone_sim.simulate_germline()
    clone_sim.simulate_clone_genealogy()
    clone_sim.sim_somatic_mutations(mut_rate=1e-6)
    assert clone_sim.n_somatic_mut > 0
    clone_sim.simulate_germline_somatic_muts()
    assert np.any(clone_sim.germline_somatic_pl != 0)
    clone_sim.simulate_clonal_germline_muts()
    assert np.any(clone_sim.germline_clone_pl != 0)


@pytest.fixture
def sim():
    return ClonalSim(seq_len=1e6, n_clones=2)


def test_create_gt_string_nan_pl(sim):
    """NaN PL (e.g. from exp underflow in phred_rescale) must not crash and should yield equal PLs."""
    nan_pl = np.array([np.nan, np.nan, np.nan])
    gt_str, *_ = sim.create_gt_string(alt_reads=0, tot_reads=0, pl=nan_pl)
    pl_field = gt_str.split(":")[4]
    assert all(v == "0" for v in pl_field.split(","))


def test_create_gt_string_inf_pl(sim):
    """Inf PL values (exp underflow path) must also yield equal PLs."""
    inf_pl = np.array([np.inf, np.inf, np.inf])
    gt_str, *_ = sim.create_gt_string(alt_reads=0, tot_reads=0, pl=inf_pl)
    pl_field = gt_str.split(":")[4]
    assert all(v == "0" for v in pl_field.split(","))


def test_vcf_output_full_sim(tmp_path):
    """Test that the VCF output makes sense."""
    clone_sim = ClonalSim(seq_len=1e6, n_clones=5)
    clone_sim.simulate_germline()
    clone_sim.simulate_clone_genealogy()
    clone_sim.sim_somatic_mutations(mut_rate=1e-6)
    assert clone_sim.n_somatic_mut > 0
    clone_sim.simulate_germline_somatic_muts()
    assert np.any(clone_sim.germline_somatic_pl != 0)
    clone_sim.simulate_clonal_germline_muts()
    assert np.any(clone_sim.germline_clone_pl != 0)
    # Setup the output VCF file and write it out
    d = tmp_path / "sim_vcf_test"
    d.mkdir()
    vcf_fp = d / "test.vcf"
    clone_sim.write_vcf(out=vcf_fp)
    # Read back the VCF file
    reread_vcf = VCF(vcf_fp)
    assert reread_vcf.contains("AD")
    assert reread_vcf.contains("PL")
    assert len(reread_vcf.samples) == 5 + 1


# ---------------------------------------------------------------------------
# geno_loglik_2d numerical equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,J", [(1, 1), (10, 5), (100, 20)])
def test_geno_loglik_2d_matches_scalar(M, J):
    """geno_loglik_2d must agree with calling geno_loglik per (i, j).

    geno_loglik accumulates in float32 (its internal ``eps``/``gl`` are
    C floats), while geno_loglik_2d uses float64 throughout.  Results
    therefore differ by at most a fraction of a Phred unit; we allow 0.01
    Phred absolute tolerance, which is irrelevant in practice since PL
    values are rounded to integers before VCF output.
    """
    rng = np.random.default_rng(0)
    tot = rng.integers(1, 30, size=(M, J)).astype(np.int64)
    alt = rng.integers(0, tot + 1, size=(M, J)).astype(np.int64)
    alt = np.minimum(alt, tot)

    expected = np.empty((M, J, 3))
    for i in range(M):
        for j in range(J):
            expected[i, j, :] = geno_loglik(int(alt[i, j]), int(tot[i, j]))

    got = np.empty((M, J, 3))
    geno_loglik_2d(alt, tot, got)

    np.testing.assert_allclose(got, expected, atol=0.01)


def test_geno_loglik_2d_zero_coverage():
    """Sites with zero coverage should produce all-zero PL vectors."""
    alt = np.zeros((5, 3), dtype=np.int64)
    tot = np.zeros((5, 3), dtype=np.int64)
    out = np.empty((5, 3, 3))
    geno_loglik_2d(alt, tot, out)
    assert np.all(out == 0.0)


def test_geno_loglik_2d_1d_reshape_matches_scalar():
    """The reshape(-1,1) + squeeze idiom used at 1-D call sites is equivalent."""
    rng = np.random.default_rng(1)
    M = 50
    tot_1d = rng.integers(1, 40, size=M).astype(np.int64)
    alt_1d = rng.integers(0, tot_1d + 1, size=M).astype(np.int64)
    alt_1d = np.minimum(alt_1d, tot_1d)

    # Assign immediately into a pre-allocated array to avoid geno_loglik's
    # stack-backed memoryview being invalidated between list elements.
    expected = np.empty((M, 3))
    for i, (a, t) in enumerate(zip(alt_1d, tot_1d)):
        expected[i, :] = geno_loglik(int(a), int(t))

    out3d = np.empty((M, 1, 3))
    geno_loglik_2d(alt_1d.reshape(-1, 1), tot_1d.reshape(-1, 1), out3d)
    got = out3d[:, 0, :]

    np.testing.assert_allclose(got, expected, atol=0.01)


# ---------------------------------------------------------------------------
# create_read_matrix correctness
# ---------------------------------------------------------------------------


def test_create_read_matrix_shape_and_values():
    """create_read_matrix must return (M_som + M_germ, J, 2) with ref+alt = tot."""
    sim = ClonalSim(seq_len=1e6, n_clones=5)
    sim.simulate_germline(seed=7)
    sim.simulate_clone_genealogy(seed=7)
    sim.sim_somatic_mutations(mut_rate=1e-6, seed=7)
    sim.simulate_clonal_germline_muts(seed=7)

    X = sim.create_read_matrix()

    expected_M = sim.n_somatic_mut + sim.n_germline_poly
    assert X.shape == (expected_M, sim.J, 2)

    # ref + alt must equal total reads for every (site, clone)
    somatic_tot = sim.somatic_tot_reads  # (n_somatic_mut, J)
    germline_tot = sim.germline_clone_tot_reads  # (n_germline_poly, J)
    expected_tot = np.vstack([somatic_tot, germline_tot])
    np.testing.assert_array_equal(X[:, :, 0] + X[:, :, 1], expected_tot)


def test_create_read_matrix_alt_values():
    """Alt-read slice of create_read_matrix must equal the stored alt arrays."""
    sim = ClonalSim(seq_len=1e6, n_clones=4)
    sim.simulate_germline(seed=99)
    sim.simulate_clone_genealogy(seed=99)
    sim.sim_somatic_mutations(mut_rate=1e-6, seed=99)
    sim.simulate_clonal_germline_muts(seed=99)

    X = sim.create_read_matrix()

    expected_alt = np.vstack([sim.somatic_alt_reads, sim.germline_clone_alt_reads])
    np.testing.assert_array_equal(X[:, :, 1], expected_alt)
