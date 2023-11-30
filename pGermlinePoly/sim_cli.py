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
    "--seqlen",
    "-l",
    required=True,
    type=int,
    default=10000000,
    help="Length of contig to simulate somatic mutations along.",
)
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
    "--germline_q",
    "-gq",
    type=float,
    default=30.0,
    help="Assumed read-quality score for germline.",
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
    "--clone_q",
    "-cq",
    type=float,
    default=30.0,
    help="Assumed read-quality score for clones.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random number seed for simulations.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="out.vcf",
    help="Output VCF file",
)
@click.option(
    "--out_tree",
    "-ot",
    required=False,
    type=str,
    default=None,
    help="Output newick file for somatic clone genealogy (unscaled).",
)
def main(
    seqlen,
    nclones,
    age,
    germline_mu,
    somatic_mu,
    mean_germline_cov,
    var_germline_cov,
    germline_q,
    mean_clone_cov,
    var_clone_cov,
    clone_q,
    seed,
    out,
    out_tree,
):
    """CLI for calculating probability of germline polymorphism from somatic clonal sequencing data."""
    clone_sim = ClonalSim(seq_len=seqlen, n_clones=nclones)
    # NOTE: should decide on the AFS format too ...
    clone_sim.simulate_germline(
        mean_coverage=mean_germline_cov,
        var_coverage=var_germline_cov,
        mut_rate=germline_mu,
        q=germline_q,
        seed=seed,
    )
    clone_sim.simulate_clone_genealogy(age=age, seed=seed)
    clone_sim.sim_somatic_mutations(
        age=age,
        mut_rate=somatic_mu,
        mean_coverage=mean_clone_cov,
        var_coverage=var_clone_cov,
        q=clone_q,
        seed=seed,
    )
    clone_sim.simulate_germline_somatic_muts(
        mean_coverage=mean_germline_cov,
        var_coverage=var_germline_cov,
        q=germline_q,
        seed=seed,
    )
    clone_sim.simulate_clonal_germline_muts(
        mean_coverage=mean_clone_cov, var_coverage=var_clone_cov, q=clone_q, seed=seed
    )
    # Setup the output VCF file and write it out
    clone_sim.write_vcf(out=out)

    # Optionally write out the tree file if it is provided
    if out_tree is not None:
        with open(out_tree, "w+") as tree_out:
            n_str = clone_sim.genealogy.newick(precision=3)
            tree_out.write(n_str)
