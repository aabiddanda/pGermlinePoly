"""CLI for karyohmm."""
import logging
import sys

import click
from cyvcf2 import VCF
from tqdm import tqdm

from pGermlinePoly import ProbGermline
from pGermlinePoly.io import *

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--vcf",
    "-v",
    required=True,
    type=click.Path(exists=True),
    help="Input VCF file.",
)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Input file detailing clone structure.",
)
@click.option(
    "--nthreads",
    "-t",
    required=False,
    type=int,
    default=1,
    help="Number of threads.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="out.vcf.gz",
    help="Output VCF file",
)
def main(vcf, config, nthreads, out):
    """CLI for calculating probability of germline polymorphism from somatic clonal sequencing data."""
    logging.info("Checking config structure ...")
    config = validate_config(config)
    logging.info("Finished config structure check!")
    logging.info(f"Starting VCF checks on {vcf}...")
    samples = config["germline"] + config["clones"]
    annotations = config["annotations"]
    cur_vcf = VCF(vcf, samples=samples)
    check_samples(cur_vcf, samples=samples)
    check_annotations(cur_vcf, annotations=annotations)
    logging.info(f"Finished VCF checks on {vcf}!")
    logging.info("Extracting data for inference...")
    germline_vcf = VCF(vcf, samples=config["germline"], threads=nthreads)
    germline_anno = create_germline_anno(germline_vcf)
    clonal_vcf = VCF(vcf, samples=config["clones"], threads=nthreads)
    clone_pl = create_clonal_pl_matrix(clonal_vcf)
    anno_vcf = VCF(vcf, samples=config["clones"], threads=nthreads)
    anno = create_anno(anno_vcf, annotations=annotations)
    print(anno.shape, germline_anno.shape)
    # this is a little strange in terms of dimensions here ...
    full_anno = np.vstack([germline_anno, anno]).T
    print(full_anno.shape)
    logging.info("Finished extracting data for inference!")
    logging.info("Starting EM-algorithm...")
    p_germline = ProbGermline(X=clone_pl, Theta=full_anno)
    p_germline.em_algo()
    logging.info("Finished EM-algorithm!")
    logging.info(f"Writing VCF output to {out} ...")
    logging.info(f"Finished writing annotated VCF to {out}!")
