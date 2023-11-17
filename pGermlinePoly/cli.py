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
    """CLI for calculating probability of germline heterozygosity.

    NOTE: we can either have a config file with a specific structure or we can feed in sample-ids via the command line.
    """
    logging.info(f"Checking config structure ...")
    config = validate_config(config)
    logging.info(f"Finished config structure check!")
    logging.info(f"Starting VCF checks on {vcf}...")
    samples = config["germline"] + config["clones"]
    cur_vcf = VCF(vcf, samples=samples, threads=nthreads)
    check_samples(cur_vcf, samples=samples)
    check_annotations(cur_vcf, annotations=annotations)
    logging.info(f"Finished quality checks on {vcf}!")
    logging.info(f"Extracting data for EM-algorithm ...")
    # TEST ...
    logging.info(f"Finished extracting data for EM-algorithm ...")
    logging.info(f"Starting EM-algorithm...")
    # p_germline = ProbGermline()
    logging.info(f"Finished EM-algorithm!")
    logging.info(f"Writing VCF output to {out} ...")
    logging.info(f"Finished writing annotated VCF to {out}!")
