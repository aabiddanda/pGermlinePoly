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
    """Validate a config-file using an underlying schema."""
    v = Validator(schema)
    with open(config_yaml_fp, "r") as stream:
        config = yaml.safe_load(stream)
        assert v.validate(config)
    return config


def check_samples(vcf, samples=[]):
    """Check that all of the requested samples are in the VCF."""
    for s in samples:
        assert s in vcf.samples


def check_annotations(vcf, annotations=["PL", "AD"]):
    """Check that the annotations are contained as site-level information as either INFO or FORMAT fields."""
    for a in annotations:
        assert vcf.contains(a)


def create_germline_anno(vcf, **kwargs):
    """Create the germline annotation from allele depths."""
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
    """Extract annotation values from VCF and transpose them."""
    total_anno = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            anno = [v.INFO[a] for a in annotations]
        else:
            anno = [np.nan for a in annotations]
        total_anno.append(anno)
    return np.vstack(total_anno)


def create_read_matrix(vcf):
    """
    Create a matrix of read-counts from clonal samples.

    The resulting matrix is a K x J x 2 matrix with the read-counts averaged across all clonal samples.
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
