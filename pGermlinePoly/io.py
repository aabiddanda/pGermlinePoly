"""Module to help with IO routines and validation."""
import numpy as np
import yaml
from cerberus import Validator
from tqdm import tqdm

germline_schema = {
    "ind": {"required": True, "type": "string"},
    "sex": {"required": True, "type": "string"},
    "age": {"required": True, "type": "int"},
    "germline": {
        "required": True,
        "type": "list",
        "schema": {"value": {"type": "string"}},
    },
    "clones": {
        "required": True,
        "type": "list",
        "schema": {"value": {"type": "string"}},
    },
}


def validate_config(config_yaml_fp, schema=germline_schema):
    """Validate a config-file using an underlying schema."""
    v = Validator(schema)
    with open(config_yaml_fp, "r") as stream:
        config = yaml.load(stream)
        assert v.validate(config)
    return config


def check_samples(vcf, samples=[]):
    """Check that all of the requested samples are in the VCF."""
    for s in samples:
        assert s in vcf.samples


def check_anno(vcf, annotations=["PL", "AD"]):
    """Check that the annotations are contained as site-level information as either INFO or FORMAT fields."""
    for a in annotations:
        assert a in vcf.contains(a)


def create_germline_anno(vcf):
    """Create the germline annotation for the clonal sequencing data.

    NOTE: currently this method only considers biallelic SNVs in the annotation model.
    NOTE: this can support more than one germline sample if available as well to improve inference.
    """
    assert vcf.contains("PL")
    germline_log_ratio = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            x = v.format("PL")
            poly_lrr = np.min(x[:, 1:-1], axis=1) - np.min(x[:, [0, -1]])
            germline_log_ratio.append(np.mean(poly_lrr))
        else:
            germline_log_ratio.append(np.nan)
    return np.array(germline_log_ratio, dtype=np.float32)


def create_clonal_pl_matrix(vcf):
    """Create the X matrix for inference from clonal samples.

    The X matrix is a K x J x 3 tensor for biallelic SNPs of the normalized PL values from GATK.
    """
    assert vcf.contains("PL")
    assert len(vcf.samples) > 1
    X = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            x = v.format("PL")[0]
            X.append(x)
        else:
            # NOTE: we could replace these with NaNs as well to signify missing data...
            X.append(np.zeros(shape=(len(vcf.samples), 3)))
    X = np.stack(X)
    return X
