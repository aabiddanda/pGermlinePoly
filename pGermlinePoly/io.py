"""Module to help with IO routines and validation."""

import numpy as np
import yaml
from cerberus import Validator
from poly_utils import logsumexp
from tqdm import tqdm

germline_schema = {
    "ind": {"required": True, "type": "string"},
    "sex": {"required": True, "type": "string", "maxlength": 1, "allowed": ["M", "F"]},
    "age": {"required": True, "type": "number", "min": 0.0},
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


def invert_pl(pl):
    """Invert the PL field to a normalized genotype log-likelihood."""
    pl = np.array(pl)
    assert np.all(pl >= 0)
    assert pl.ndim == 1
    assert pl.size > 1
    p_gt = pl / -10.0
    p_gt = np.nan_to_num(p_gt)
    p_gt /= np.log10(np.e)
    p_gt = p_gt - logsumexp(p_gt)
    return p_gt


def create_germline_anno_gl(vcf):
    """Create the germline annotation for the clonal sequencing data.

    NOTE: currently this method only considers biallelic SNVs in the annotation model.
    NOTE: this can support more than one germline sample if available as well to improve inference.
    """
    assert vcf.contains("PL")
    germline_log_ratio = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            x = v.format("PL")
            if x.ndim == 1:
                pl_x = invert_pl(x)
                poly_lrr = logsumexp(pl_x[1:-1]) - logsumexp(pl_x[[0, -1]])
                germline_log_ratio.append(poly_lrr)
            else:
                poly_lrr = np.zeros(x.shape[0])
                for i in range(x.shape[0]):
                    pl_x = invert_pl(x[i])
                    poly_lrr[i] = logsumexp(pl_x[1:-1]) - logsumexp(pl_x[[0, -1]])
                germline_log_ratio.append(np.mean(poly_lrr))
        else:
            germline_log_ratio.append(np.nan)
    germline_log_ratio = np.array(germline_log_ratio, dtype=np.float32)
    assert germline_log_ratio.ndim == 1
    return germline_log_ratio


def create_anno(vcf, annotations=[]):
    """Extract annotation values from VCF and transpose them."""
    total_anno = []
    for v in tqdm(vcf):
        if v.is_snp and (len(v.ALT) == 1):
            anno = [v.INFO[a] for a in annotations]
        else:
            anno = [np.nan for a in annotations]
        total_anno.append(anno)
    return np.vstack(total_anno).T


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
    X = np.stack(X)
    return X
