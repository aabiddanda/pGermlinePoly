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


def create_germline_anno(vcf, germline_sample, biallelic_snps=True):
    """Create the germline annotation for the clonal sequencing data."""
    assert germline_sample in vcf.samples
    assert vcf.contains("PL")
    germline_log_ratio = []
    for v in tqdm(vcf):
        if biallelic_snps:
            if v.is_snp and (len(v.ALT) == 1):
                x = np.array(vcf.format("PL")[0])
                germline_log_ratio.append(np.sum(x[1:-1]) - (x[0] + x[-1]))
            else:
                germline_log_ratio.append(np.nan)
        else:
            raise NotImplementedError(
                "Current implementation of pGermlinePoly focuses on biallelic SNVs only."
            )


def create_clonal_pl_matrix(vcf, biallelic_snps=True):
    """Create the X matrix for inference from clonal samples.

    The X matrix is a K x J x 3 tensor for biallelic SNPs.
    """
    assert vcf.contains("PL")
    X = []
    for v in tqdm(vcf):
        if biallelic_snps:
            if v.is_snp and (len(v.ALT) == 1):
                x = vcf.format("PL")[0]
                X.append(x)
            else:
                # NOTE: we could replace this with NaNs as well to signify missing data...
                X.append(np.zeros(shape=(len(vcf.samples), 3)))
    X = np.hstack(X)
    return X
