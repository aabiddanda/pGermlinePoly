"""Module to help with IO routines and validation."""

import numpy as np
import yaml
from cerberus import Validator
from poly_utils import geno_loglik
from tqdm import tqdm

germline_schema = {
    "ind": {"required": True, "type": "string"},
    "sex": {"required": True, "type": "string", "maxlength": 1, "allowed": ["M", "F"]},
    "age": {"required": True, "type": "number", "min": 0.0},
    "germline": {"required": False, "type": "list", "schema": {"type": "string"}},
    "clones": {"required": True, "type": "list", "schema": {"type": "string"}},
    "annotations": {"required": True, "type": "list"},
}


def validate_config(config_yaml_fp, schema=germline_schema):
    """Validate a YAML configuration file against the germline schema.

    Parameters
    ----------
    config_yaml_fp : str
        Path to the YAML configuration file.
    schema : dict, optional
        Cerberus schema to validate against. Default is ``germline_schema``.

    Returns
    -------
    dict
        Parsed and validated configuration dictionary.

    Raises
    ------
    AssertionError
        If the configuration does not conform to the schema.
    """
    v = Validator(schema)
    with open(config_yaml_fp, "r") as stream:
        config = yaml.safe_load(stream)
        assert v.validate(config)
    return config


def check_samples(vcf, samples=[]):
    """Assert that all requested sample names are present in the VCF.

    Parameters
    ----------
    vcf : cyvcf2.VCF
        Opened VCF object with a ``samples`` attribute.
    samples : list of str, optional
        Sample names to verify. Default is an empty list.

    Raises
    ------
    AssertionError
        If any name in ``samples`` is not found in ``vcf.samples``.
    """
    for s in samples:
        assert s in vcf.samples


def check_annotations(vcf, annotations=["PL", "AD"]):
    """Assert that required annotation fields are declared in the VCF header.

    Checks both INFO and FORMAT fields via ``vcf.contains``.

    Parameters
    ----------
    vcf : cyvcf2.VCF
        Opened VCF object.
    annotations : list of str, optional
        Annotation field IDs to verify. Default is ``["PL", "AD"]``.

    Raises
    ------
    AssertionError
        If any field ID in ``annotations`` is not declared in the VCF header.
    """
    for a in annotations:
        assert vcf.contains(a)


def create_germline_anno(vcf, **kwargs):
    """Extract per-site germline heterozygote log-likelihoods from a germline VCF.

    Iterates over biallelic SNPs, reads the AD FORMAT field of the first
    sample (assumed to be the germline sample), computes Phred-scaled
    genotype likelihoods via ``geno_loglik``, and returns the heterozygote
    PL value (index 1) for each site.

    Parameters
    ----------
    vcf : cyvcf2.VCF
        Opened VCF with an "AD" FORMAT field. The first sample is treated
        as the germline reference.
    **kwargs
        Additional keyword arguments forwarded to ``geno_loglik`` (e.g., ``q``).

    Returns
    -------
    numpy.ndarray
        Heterozygote genotype log-likelihoods for each biallelic SNP,
        shape (M,), dtype float64.

    Raises
    ------
    AssertionError
        If the VCF does not contain the "AD" FORMAT field.
    """
    assert vcf.contains("AD")
    germline_anno = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            x = v.format("AD")
            assert x.ndim == 2
            gl = geno_loglik(x[0][1], x[0][0], **kwargs)
            germline_anno.append(gl[1])
    germline_anno = np.array(germline_anno).astype(np.float64)
    return germline_anno


def create_anno(vcf, annotations=[]):
    """Extract INFO annotation values for all variants in a VCF.

    Iterates over all variants, collecting the requested INFO field values
    for biallelic SNPs. Non-SNP or multiallelic sites receive NaN for all
    requested annotations.

    Parameters
    ----------
    vcf : cyvcf2.VCF
        Opened VCF object.
    annotations : list of str, optional
        INFO field IDs to extract. Default is an empty list.

    Returns
    -------
    numpy.ndarray
        Annotation matrix of shape (N, len(annotations)), where N is the
        total number of variants iterated.
    """
    total_anno = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            anno = [v.INFO.get(a) for a in annotations]
        else:
            anno = [np.nan for a in annotations]
        total_anno.append(anno)
    return np.vstack(total_anno)


def create_read_matrix(vcf):
    """Build a read-count matrix from the AD FORMAT field of a clonal VCF.

    Iterates over all variants. For biallelic SNPs the allele depth (AD)
    matrix is stacked; non-SNP or multiallelic records are represented as
    rows of zeros.

    Parameters
    ----------
    vcf : cyvcf2.VCF
        Opened VCF object containing an "AD" FORMAT field with at least two
        samples (clones).

    Returns
    -------
    numpy.ndarray
        Integer read-count array of shape (M, J, 2), where M is the number
        of variants, J is the number of samples, and the last dimension
        holds [ref_reads, alt_reads] per sample.

    Raises
    ------
    AssertionError
        If the VCF does not contain the "AD" FORMAT field or has fewer than
        two samples.
    """
    assert vcf.contains("AD")
    assert len(vcf.samples) > 1
    X = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            x = v.format("AD")
            assert x.ndim == 2
            A = x
            X.append(A)
        else:
            # NOTE: this could be replaced with nans as well...
            X.append(np.zeros(shape=(len(vcf.samples), 2)))
    X = np.stack(X).astype(dtype="int")
    return X
