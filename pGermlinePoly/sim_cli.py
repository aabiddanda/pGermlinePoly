"""CLI for simulation of neutral clonal sequencing data."""
import logging
import sys

import click
from cyvcf2 import VCF
from tqdm import tqdm

from pGermlinePoly import ClonalSim

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--nclones",
    "-j",
    required=True,
    type=int,
    default=5,
    help="Number of sampled clones.",
)
@click.option(
    "--age",
    "-a",
    required=True,
    type=float,
    default=30.0,
    help="Age of individual from clonal sampling.",
)
@click.option(
    "--germline_mu",
    "-mu",
    type=float,
    default=1.2e-8,
    help="Germline mutation rate (per bp / per generation)",
)
@click.option(
    "--somatic_mu",
    "-smu",
    type=float,
    default=1.2e-9,
    help="Somatic mutation rate (per bp / per year)",
)
@click.option(
    "--mean_germline_cov",
    "-g",
    type=float,
    default=15.0,
    help="Mean coverage in germline sample.",
)
@click.option(
    "--var_germline_cov",
    "-v",
    type=float,
    default=5.0,
    help="Variance in germline coverage.",
)
@click.option(
    "--mean_clone_cov",
    "-gc",
    type=float,
    default=15.0,
    help="Mean coverage in clone sequencing depth.",
)
@click.option(
    "--var_clone_cov",
    "-vc",
    type=float,
    default=5.0,
    help="Variance in clone sequencing depth.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="out.vcf.gz",
    help="Output VCF file",
)
def main(
    nclones,
    age,
    germline_mu,
    somatic_mu,
    mean_germline_cov,
    var_germline_cov,
    mean_clone_cov,
    var_clone_cov,
    out,
):
    """CLI for calculating probability of germline polymorphism from somatic clonal sequencing data."""
    pass
