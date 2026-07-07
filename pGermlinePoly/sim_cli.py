"""CLI for simulation of neutral clonal sequencing data."""

import logging

import rich_click as click
import yaml

from pGermlinePoly import ClonalSim


def setup_logging(logfile=None):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if logfile:
        handler = logging.FileHandler(logfile)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        root.addHandler(handler)
    elif not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        root.addHandler(handler)


@click.command(
    help="Simulation engine for neutral clonal sequencing data with a germline sample.",
    context_settings=dict(show_default=True),
)
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
    "--afs_alpha",
    type=float,
    default=0.31699444395046117,
    help="Estimate of alpha parameter for allele frequency - default from NFE AF in gnomAD v3",
)
@click.option(
    "--afs_beta",
    type=float,
    default=6.067159920986527,
    help="Estimate of beta parameter for allele frequency - default from NFE AF in gnomAD v3",
)
@click.option(
    "--germline_het",
    "-het",
    type=float,
    default=1e-3,
    help="Heterozygosity rate (per bp)",
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
    "--sd_germline_cov",
    "-v",
    type=float,
    default=5.0,
    help="Standard deviation in germline coverage.",
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
    "--sd_clone_cov",
    "-vc",
    type=float,
    default=5.0,
    help="Standard deviation in clone sequencing depth.",
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
@click.option(
    "--out_config",
    "-oc",
    required=False,
    type=str,
    default=None,
    help="Output yaml-based config file for applying inference post-hoc.",
)
@click.option(
    "--logfile",
    required=False,
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Write log messages to this file instead of stderr.",
)
def main(
    seqlen,
    nclones,
    age,
    afs_alpha,
    afs_beta,
    germline_het,
    germline_mu,
    somatic_mu,
    mean_germline_cov,
    sd_germline_cov,
    germline_q,
    mean_clone_cov,
    sd_clone_cov,
    clone_q,
    seed,
    out,
    out_tree,
    out_config,
    logfile,
):
    """CLI for calculating probability of germline polymorphism from somatic clonal sequencing data."""
    setup_logging(logfile)
    logging.info(
        f"Setting up somatic simulation object for {nclones} clones over {seqlen} basepairs"
    )
    clone_sim = ClonalSim(seq_len=seqlen, n_clones=nclones)
    logging.info(
        f"Simulating germline variants with {mean_germline_cov} ({sd_germline_cov}) coverage ..."
    )
    clone_sim.simulate_germline(
        afs=[afs_alpha, afs_beta],
        het_rate=germline_het,
        mean_coverage=mean_germline_cov,
        sd_coverage=sd_germline_cov,
        mut_rate=germline_mu,
        q=germline_q,
        seed=seed,
    )
    logging.info(f"Simulated  {clone_sim.n_germline_poly} germline variants ... ")
    logging.info(f"Simulating a clonal genealogy for {nclones} clones ...")
    clone_sim.simulate_clone_genealogy(age=age, seed=seed)
    clone_sim.sim_somatic_mutations(
        age=age,
        mut_rate=somatic_mu,
        mean_coverage=mean_clone_cov,
        sd_coverage=sd_clone_cov,
        q=clone_q,
        seed=seed,
    )
    logging.info(
        f"Simulated {clone_sim.n_somatic_mut} somatic mutations for an individual of age {age} with {mean_clone_cov} ({sd_clone_cov}) coverage ...!"
    )
    logging.info(f"Filling in germline mutation status for {nclones} clones!")
    clone_sim.simulate_germline_somatic_muts(
        mean_coverage=mean_germline_cov,
        sd_coverage=sd_germline_cov,
        q=germline_q,
        seed=seed,
    )
    logging.info("Filling in somatic mutation status for the germline sample!")
    clone_sim.simulate_clonal_germline_muts(
        mean_coverage=mean_clone_cov, sd_coverage=sd_clone_cov, q=clone_q, seed=seed
    )
    # Setup the output VCF file and write it out
    logging.info(f"Writing out the VCF file to {out}")
    clone_sim.write_vcf(out=out)
    # Optionally write out the tree file if it is provided
    if out_tree is not None:
        logging.info(
            f"Writing out the (unscaled) somatic cellular phylogeny to {out_tree}!"
        )
        with open(out_tree, "w+") as tree_out:
            n_str = clone_sim.genealogy.newick(precision=3)
            tree_out.write(n_str)
    if out_config is not None:
        logging.info(f"Writing out the config for inference to {out_config}!")
        data = dict(
            ind="IndA",
            age=age,
            sex="M",
            germline=["Agermline"],
            clones=[f"Aclone{i}" for i in range(nclones)],
            annotations=["ExternalAF"],
        )
        with open(out_config, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
