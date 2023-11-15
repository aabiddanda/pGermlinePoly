"""CLI for karyohmm."""
import logging
import sys

import click
from cyvcf2 import VCF
from tqdm import tqdm

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
    default=4,
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
def main(vcf, nthreads, out):
    """Karyohmm CLI."""
    logging.info(f"Starting to read input data {vcf}.")
    logging.info(f"Finished reading in {vcf}.")
    logging.info("Finished karyohmm analysis!")
