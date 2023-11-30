import numpy as np
import pytest
from cyvcf2 import VCF

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
