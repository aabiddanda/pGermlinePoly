"""Module to help with IO routines and validation."""
import numpy as np
import yaml
from cerberus import Validator
from tqdm import tqdm

germline_schema = {
    "ind": {"required": True, "type": "string"},
    "sex": {"required": True, "type": "string", "maxlength": 1, "allowed": ["M", "F"]},
    "age": {"required": True, "type": "integer", "min": 0},
    "germline": {"required": True, "type": "list", "schema": {"type": "string"}},
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


def create_anno(vcf, annotations=[]):
    """Extract annotation values from VCF."""
    total_anno = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            anno = [v.INFO[a] for a in annotations]
        else:
            anno = [np.nan for a in annotations]
        total_anno.append(anno)
    return np.hstack(total_anno)


def create_clonal_pl_matrix(vcf):
    """Create the X matrix for inference from clonal samples.

    The X matrix is a K x J x 3 tensor for biallelic SNPs of the normalized PL values from GATK.
    """
    assert vcf.contains("PL")
    assert len(vcf.samples) > 1
    X = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            x = v.format("PL")
            X.append(x)
        else:
            # NOTE: we could replace these with NaNs as well to signify missing data...
            X.append(np.zeros(shape=(len(vcf.samples), 3)))
    X = np.stack(X)
    return X
