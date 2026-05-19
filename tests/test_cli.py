"""End-to-end CLI integration tests for pGermlinePoly."""

import math
import re
import types

import pytest
import yaml
from click.testing import CliRunner
from cyvcf2 import VCF

from pGermlinePoly import ClonalSim
from pGermlinePoly.cli import main

runner = CliRunner()


# ---------------------------------------------------------------------------
# Session-scoped fixture: simulate a VCF once for the whole test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sim_vcf_paths(tmp_path_factory):
    """Generate a realistic VCF with germline and somatic variants via ClonalSim."""
    d = tmp_path_factory.mktemp("sim_vcf")
    vcf_fp = d / "sim.vcf"

    sim = ClonalSim(seq_len=1e6, n_clones=5)
    sim.simulate_germline(seed=42)
    sim.simulate_clone_genealogy(age=80, seed=42)
    sim.sim_somatic_mutations(age=80, mut_rate=1e-7, seed=42)
    assert sim.n_somatic_mut > 0, (
        "No somatic mutations in fixture — increase mut_rate or seq_len"
    )
    sim.simulate_germline_somatic_muts()
    sim.simulate_clonal_germline_muts()
    sim.write_vcf(out=str(vcf_fp))

    cfg_fp = d / "config.yaml"
    cfg_fp.write_text(yaml.dump({
        "ind": "IndA", "age": 50, "sex": "M",
        "clones": [f"Aclone{i}" for i in range(5)],
        "annotations": ["ExternalAF"],
    }))

    cfg_germline_fp = d / "config_germline.yaml"
    cfg_germline_fp.write_text(yaml.dump({
        "ind": "IndA", "age": 50, "sex": "M",
        "germline": ["Agermline"],
        "clones": [f"Aclone{i}" for i in range(5)],
        "annotations": ["ExternalAF"],
    }))

    return types.SimpleNamespace(
        vcf_fp=vcf_fp,
        cfg_fp=cfg_fp,
        cfg_germline_fp=cfg_germline_fp,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(args):
    """Invoke the CLI, converting all path-like args to strings."""
    return runner.invoke(main, [str(a) for a in args], catch_exceptions=True)


# ---------------------------------------------------------------------------
# Happy-path integration tests
# ---------------------------------------------------------------------------

def test_cli_em_default(sim_vcf_paths, tmp_path):
    """EM algorithm (default) runs end-to-end and writes expected INFO fields."""
    out_fp = tmp_path / "out.vcf"
    result = _run(["--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp, "-o", out_fp])
    assert result.exit_code == 0, result.output
    content = out_fp.read_text()
    assert "ppGermlinePoly" in content
    assert "mleVAF" in content
    assert "##lambda_ExternalAF=" in content
    assert "##kappa_hat=" in content
    assert "##pGermlinePoly=run" in content
    # Optional fields must be absent when flags not passed
    assert "lrtGermlinePoly" not in content
    assert "lodMutect" not in content
    assert "rhobeta" not in content



def test_cli_with_germline_vcf(sim_vcf_paths, tmp_path):
    """Germline annotation path adds lambda_germline weight to VCF header."""
    out_fp = tmp_path / "out.vcf"
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp,
        "--config", sim_vcf_paths.cfg_germline_fp,
        "--germline_vcf", sim_vcf_paths.vcf_fp,
        "-o", out_fp,
    ])
    assert result.exit_code == 0, result.output
    content = out_fp.read_text()
    assert "##lambda_germline=" in content
    assert "##lambda_ExternalAF=" in content


def test_cli_lrt_flag(sim_vcf_paths, tmp_path):
    """--lrt adds lrtGermlinePoly to INFO header and data records."""
    out_fp = tmp_path / "out.vcf"
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp,
        "--lrt", "-o", out_fp,
    ])
    assert result.exit_code == 0, result.output
    content = out_fp.read_text()
    assert "lrtGermlinePoly" in content
    data_lines = [l for l in content.splitlines() if not l.startswith("#")]
    assert any("lrtGermlinePoly=" in l for l in data_lines)


def test_cli_mutect2_flag(sim_vcf_paths, tmp_path):
    """--mutect2 adds lodMutect to INFO header and data records."""
    out_fp = tmp_path / "out.vcf"
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp,
        "--mutect2", "-o", out_fp,
    ])
    assert result.exit_code == 0, result.output
    content = out_fp.read_text()
    assert "lodMutect" in content
    data_lines = [l for l in content.splitlines() if not l.startswith("#")]
    assert any("lodMutect=" in l for l in data_lines)


def test_cli_betabinomial_flag(sim_vcf_paths, tmp_path):
    """--betabinomial adds rhobeta to INFO header and data records."""
    out_fp = tmp_path / "out.vcf"
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp,
        "--betabinomial", "-o", out_fp,
    ])
    assert result.exit_code == 0, result.output
    content = out_fp.read_text()
    assert "rhobeta" in content
    data_lines = [l for l in content.splitlines() if not l.startswith("#")]
    assert any("rhobeta=" in l for l in data_lines)


def test_cli_all_optional_flags(sim_vcf_paths, tmp_path):
    """All three optional annotation flags coexist without interference."""
    out_fp = tmp_path / "out.vcf"
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp,
        "--lrt", "--mutect2", "--betabinomial", "-o", out_fp,
    ])
    assert result.exit_code == 0, result.output
    content = out_fp.read_text()
    for field in ("lrtGermlinePoly", "lodMutect", "rhobeta"):
        assert field in content, f"Expected INFO field '{field}' missing from output"


def test_cli_output_to_file(sim_vcf_paths, tmp_path):
    """Explicit -o path creates a non-empty output file."""
    out_fp = tmp_path / "explicit_out.vcf"
    result = _run(["--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp, "-o", out_fp])
    assert result.exit_code == 0
    assert out_fp.exists()
    assert out_fp.stat().st_size > 0
    assert "ppGermlinePoly" in out_fp.read_text()


# ---------------------------------------------------------------------------
# Parametrized optimizer tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ["L-BFGS-B", "Powell", "Nelder-Mead"])
def test_cli_algo_parametrized(sim_vcf_paths, tmp_path, algo):
    """All three --algo choices produce valid output with ppGermlinePoly."""
    out_fp = tmp_path / f"out_{algo.replace('-', '_')}.vcf"
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp,
        "--algo", algo, "-o", out_fp,
    ])
    assert result.exit_code == 0, f"algo={algo} failed: {result.output}"
    assert "ppGermlinePoly" in out_fp.read_text()


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_cli_missing_vcf(sim_vcf_paths, tmp_path):
    """Non-existent VCF path causes non-zero exit (Click Path(exists=True))."""
    result = _run([
        "--vcf", "/nonexistent/path.vcf",
        "--config", sim_vcf_paths.cfg_fp,
        "-o", tmp_path / "out.vcf",
    ])
    assert result.exit_code != 0


def test_cli_missing_config(sim_vcf_paths, tmp_path):
    """Non-existent config path causes non-zero exit."""
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp,
        "--config", "/nonexistent/config.yaml",
        "-o", tmp_path / "out.vcf",
    ])
    assert result.exit_code != 0


def test_cli_bad_config_sex(sim_vcf_paths, tmp_path):
    """Config with invalid sex value fails validate_config."""
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(yaml.dump({
        "ind": "IndA", "age": 50, "sex": "X",
        "clones": ["Aclone0"], "annotations": ["ExternalAF"],
    }))
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp, "--config", bad_cfg,
        "-o", tmp_path / "out.vcf",
    ])
    assert result.exit_code != 0


def test_cli_germline_config_no_germline_arg(sim_vcf_paths, tmp_path):
    """Config with germline key but no --germline_vcf arg causes non-zero exit."""
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp,
        "--config", sim_vcf_paths.cfg_germline_fp,
        "-o", tmp_path / "out.vcf",
    ])
    assert result.exit_code != 0


def test_cli_wrong_sample(sim_vcf_paths, tmp_path):
    """Config naming a sample absent from the VCF fails check_samples."""
    bad_cfg = tmp_path / "bad_samples.yaml"
    bad_cfg.write_text(yaml.dump({
        "ind": "IndA", "age": 50, "sex": "M",
        "clones": ["Bclone0", "Bclone1"],
        "annotations": ["ExternalAF"],
    }))
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp, "--config", bad_cfg,
        "-o", tmp_path / "out.vcf",
    ])
    assert result.exit_code != 0


def test_cli_missing_annotation(sim_vcf_paths, tmp_path):
    """Config requesting an annotation absent from the VCF fails check_annotations."""
    bad_cfg = tmp_path / "bad_anno.yaml"
    bad_cfg.write_text(yaml.dump({
        "ind": "IndA", "age": 50, "sex": "M",
        "clones": [f"Aclone{i}" for i in range(5)],
        "annotations": ["NONEXISTENT_FIELD"],
    }))
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp, "--config", bad_cfg,
        "-o", tmp_path / "out.vcf",
    ])
    assert result.exit_code != 0


def test_cli_invalid_algo(sim_vcf_paths, tmp_path):
    """Unknown --algo value is rejected by Click's Choice validation."""
    result = _run([
        "--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp,
        "--algo", "BFGS", "-o", tmp_path / "out.vcf",
    ])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Structural VCF output tests
# ---------------------------------------------------------------------------

def test_cli_output_vcf_parseable(sim_vcf_paths, tmp_path):
    """Output is a well-formed VCF with ppGermlinePoly and mleVAF declared in header."""
    out_fp = tmp_path / "out.vcf"
    result = _run(["--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp, "-o", out_fp])
    assert result.exit_code == 0
    vcf = VCF(str(out_fp))
    assert vcf.contains("ppGermlinePoly"), "ppGermlinePoly not declared in VCF header"
    assert vcf.contains("mleVAF"), "mleVAF not declared in VCF header"


def test_cli_all_records_have_pp(sim_vcf_paths, tmp_path):
    """Every output record has a non-NaN ppGermlinePoly INFO value."""
    out_fp = tmp_path / "out.vcf"
    result = _run(["--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp, "-o", out_fp])
    assert result.exit_code == 0
    records = list(VCF(str(out_fp)))
    assert len(records) > 0, "Output VCF has no records"
    for v in records:
        pp = v.INFO.get("ppGermlinePoly")
        assert pp is not None, f"Record at {v.POS} missing ppGermlinePoly"
        assert not math.isnan(float(pp)), f"ppGermlinePoly is NaN at position {v.POS}"


def test_cli_mlevaf_colon_format(sim_vcf_paths, tmp_path):
    """mleVAF INFO value follows the lo:mid:hi colon-delimited format."""
    out_fp = tmp_path / "out.vcf"
    result = _run(["--vcf", sim_vcf_paths.vcf_fp, "--config", sim_vcf_paths.cfg_fp, "-o", out_fp])
    assert result.exit_code == 0
    data_lines = [l for l in out_fp.read_text().splitlines() if not l.startswith("#")]
    assert len(data_lines) > 0
    mlevaf_re = re.compile(r"mleVAF=([^;\s]+)")
    for line in data_lines:
        m = mlevaf_re.search(line)
        assert m is not None, f"mleVAF missing from record: {line[:80]}"
        parts = m.group(1).split(":")
        assert len(parts) == 3, f"mleVAF does not have 3 colon-separated parts: {m.group(1)}"
        for p in parts:
            assert p, f"Empty mleVAF component in: {m.group(1)}"
