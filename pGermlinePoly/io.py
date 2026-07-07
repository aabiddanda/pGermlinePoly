"""Module to help with IO routines and validation."""

import numpy as np
import yaml
from cerberus import Validator
from poly_utils import geno_loglik
from tqdm import tqdm

# Supported per-annotation transforms.  Values are applied element-wise;
# domain-clipping keeps 0 and negative inputs from producing -inf / NaN.
SUPPORTED_TRANSFORMS = {
    "log10": lambda x: np.log10(np.maximum(x, 1e-10)),
    "sqrt": lambda x: np.sqrt(np.maximum(x, 0.0)),
}

germline_schema = {
    "ind": {"required": True, "type": "string"},
    "sex": {"required": True, "type": "string", "maxlength": 1, "allowed": ["M", "F"]},
    "age": {"required": True, "type": "number", "min": 0.0},
    "germline": {"required": False, "type": "list", "schema": {"type": "string"}},
    "clones": {"required": True, "type": "list", "schema": {"type": "string"}},
    "annotations": {
        "required": True,
        "type": "list",
        "schema": {
            "anyof": [
                {"type": "string"},
                {
                    "type": "dict",
                    "schema": {
                        "field": {"required": True, "type": "string"},
                        "transform": {
                            "type": "string",
                            "allowed": list(SUPPORTED_TRANSFORMS),
                        },
                        "is_af": {"type": "boolean", "required": False},
                    },
                },
            ]
        },
    },
}


def parse_annotation(entry):
    """Return ``(field_name, transform_fn)`` from a string or dict annotation entry.

    Parameters
    ----------
    entry : str or dict
        Either a plain INFO field name (string) or a dict with keys ``"field"``
        (required) and ``"transform"`` (optional, one of ``SUPPORTED_TRANSFORMS``).

    Returns
    -------
    field_name : str
        INFO field name to extract from the VCF.
    transform_fn : callable or None
        Function to apply element-wise to the extracted column, or None.
    """
    if isinstance(entry, str):
        return entry, None
    return entry["field"], SUPPORTED_TRANSFORMS.get(entry.get("transform"))


def is_af_annotation(entry):
    """Return True if the annotation entry is flagged as a population allele frequency.

    AF annotations are reflected (AF → 1−AF) for sites where
    ``reorient_to_minor_allele`` swapped ref/alt, so the annotation continues
    to describe the minor allele.  Only dict entries with ``is_af: true``
    qualify; plain string entries always return False.

    Parameters
    ----------
    entry : str or dict
        Annotation entry as accepted by :func:`parse_annotation`.

    Returns
    -------
    bool
        True if ``entry`` is a dict with ``is_af: true``, False otherwise.
    """
    if isinstance(entry, str):
        return False
    return bool(entry.get("is_af", False))


def annotation_transform_name(entry):
    """Return the transform name string for an annotation entry, or None.

    Useful when the transform name (rather than the callable) is needed —
    for example to invert a transform before reflecting an allele frequency.

    Parameters
    ----------
    entry : str or dict
        Annotation entry as accepted by :func:`parse_annotation`.

    Returns
    -------
    str or None
        One of the keys in :data:`SUPPORTED_TRANSFORMS` if a transform was
        specified, otherwise None.
    """
    if isinstance(entry, str):
        return None
    return entry.get("transform")


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

    Checks both INFO and FORMAT fields via ``vcf.contains``.  Each entry in
    ``annotations`` may be a plain string (field name) or a dict with a
    ``"field"`` key (see :func:`parse_annotation`).

    Parameters
    ----------
    vcf : cyvcf2.VCF
        Opened VCF object.
    annotations : list of str or dict, optional
        Annotation entries to verify. Default is ``["PL", "AD"]``.

    Raises
    ------
    AssertionError
        If any field ID in ``annotations`` is not declared in the VCF header.
    """
    for a in annotations:
        field, _ = parse_annotation(a)
        assert vcf.contains(field)


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
    requested annotations.  Per-annotation transforms (e.g. ``log10``,
    ``sqrt``) are applied column-wise after extraction; NaN values pass
    through unchanged and can be imputed later via
    :meth:`~pGermlinePoly.ProbGermline.impute_anno`.

    Parameters
    ----------
    vcf : cyvcf2.VCF
        Opened VCF object.
    annotations : list of str or dict, optional
        Annotation entries.  Each entry is either a plain INFO field name
        (string) or a dict ``{"field": name, "transform": "log10"|"sqrt"}``.
        See :func:`parse_annotation`. Default is an empty list.

    Returns
    -------
    numpy.ndarray
        Float64 annotation matrix of shape (N, len(annotations)), where N is
        the total number of variants iterated.
    """
    parsed = [parse_annotation(a) for a in annotations]
    total_anno = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            row = [v.INFO.get(field, np.nan) for field, _ in parsed]
        else:
            row = [np.nan for _ in parsed]
        total_anno.append(row)
    arr = np.vstack(total_anno).astype(np.float64)
    for col_idx, (_, transform_fn) in enumerate(parsed):
        if transform_fn is not None:
            arr[:, col_idx] = transform_fn(arr[:, col_idx])
    return arr


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
