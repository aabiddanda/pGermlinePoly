"""CLI for pGermlinePoly Estimation in VCF."""

import logging

import rich_click as click
import numpy as np
from tqdm import tqdm
from cyvcf2 import VCF, Writer
import sys

from pGermlinePoly import ProbGermline, MutectLOD, BetaOverdispersion
from pGermlinePoly.io import (
    validate_config,
    check_samples,
    check_annotations,
    create_germline_anno,
    create_anno,
    create_read_matrix,
)

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command(
    help=(
        "Inference of annotation-informed probability of germline polymorphism "
        "from somatic clonal sequencing data using an EM algorithm that jointly "
        "estimates logistic annotation weights (lambda) and a Beta-Binomial error "
        "concentration (kappa)."
    ),
    context_settings=dict(show_default=True),
)
@click.option(
    "--vcf",
    "-v",
    required=True,
    type=click.Path(exists=True),
    help="Input VCF file for all clones.",
)
@click.option(
    "--germline_vcf",
    "-g",
    required=False,
    type=click.Path(exists=True),
    help="Input VCF for germline sample.",
)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Input config file detailing clonal structure.",
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
    "--algo",
    "-a",
    required=False,
    default="L-BFGS-B",
    type=click.Choice(["L-BFGS-B", "Powell", "Nelder-Mead"], case_sensitive=True),
    help="Optimization algorithm for the EM M-step.",
)
@click.option(
    "--eps",
    "-e",
    required=False,
    default=1e-3,
    type=float,
    help="Error rate for read-level alleles.",
)
@click.option(
    "--delta",
    "-d",
    required=False,
    default=1e-4,
    type=float,
    help="EM convergence threshold (absolute change in log-likelihood).",
)
@click.option(
    "--em",
    required=False,
    default=False,
    is_flag=True,
    type=bool,
    help="Run the EM algorithm to estimate annotation weights (lambda) and Beta-Binomial concentration (kappa).",
)
@click.option(
    "--lrt",
    required=False,
    default=False,
    is_flag=True,
    type=bool,
    help="Frequentist likelihood ratio test testing deviation from germline heterozygote.",
)
@click.option(
    "--mutect2",
    required=False,
    default=False,
    is_flag=True,
    type=bool,
    help="LOD Score from Mutect2.",
)
@click.option(
    "--betabinomial",
    required=False,
    default=False,
    is_flag=True,
    type=bool,
    help="Implement the Beta-Binomial overdispersion method from Spencer-Chapman et al.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="-",
    help="Output VCF file (defaults to stdout)",
)
def main(
    vcf,
    germline_vcf,
    config,
    nthreads,
    algo,
    eps,
    delta,
    em,
    lrt,
    mutect2,
    betabinomial,
    out,
):
    """Run the pGermlinePoly inference pipeline on an input VCF.

    Validates the config and VCF, extracts read counts and annotations,
    runs the EM algorithm to estimate logistic annotation weights (lambda)
    and the Beta-Binomial concentration (kappa), then writes an annotated
    output VCF with per-site ``ppGermlinePoly`` and ``mleVAF`` INFO fields.
    Optional flags add ``lrtGermlinePoly``, ``lodMutect``, and ``rhobeta``.
    """
    if not any([em, lrt, mutect2, betabinomial]):
        raise click.UsageError(
            "At least one of --em, --lrt, --mutect2, or --betabinomial must be specified."
        )
    logging.info("Checking config structure ...")
    config = validate_config(config)
    logging.info("Finished config structure check!")
    logging.info(f"Starting VCF checks on {vcf}...")
    samples = config["clones"]
    annotations = config["annotations"]
    cur_vcf = VCF(vcf, samples=samples)
    check_samples(cur_vcf, samples=samples)
    check_annotations(cur_vcf, annotations=annotations)
    logging.info(f"Finished VCF checks on {vcf}!")
    logging.info(f"Extracting read data from {vcf} for inference...")
    clonal_vcf = VCF(vcf, samples=config["clones"], threads=nthreads)
    clone_reads = create_read_matrix(clonal_vcf)
    logging.info(
        f"Extracted read data from {vcf} across {len(config['clones'])} clones!"
    )
    logging.info(f"Extracting annotation data for inference from {vcf}...")
    anno_vcf = VCF(vcf, samples=config["clones"], threads=nthreads)
    anno = create_anno(anno_vcf, annotations=annotations)
    logging.info(f"Extracted annotation data from {vcf} across {len(annotations)}")
    if "germline" in config:
        assert germline_vcf is not None
        # NOTE: this could be provided as an alternative VCF as well ...
        logging.info(
            f"Extracting germline annotation from {germline_vcf} for inference..."
        )
        germ_vcf_check = VCF(germline_vcf, samples=config["germline"])
        check_samples(germ_vcf_check, samples=config["germline"])
        germ_vcf = VCF(germline_vcf, samples=config["germline"], threads=nthreads)
        germline_anno = create_germline_anno(germ_vcf)
        logging.info(f"Extracted germline annotation from {vcf} for inference!")
        full_anno = np.hstack([germline_anno[:, None], anno])
    else:
        full_anno = anno

    p_germline = ProbGermline(X=clone_reads, Theta=full_anno, mu=eps)
    logging.info("Imputing missing annotations...")
    p_germline.impute_anno()
    logging.info("Finished imputing missing annotations!")
    if em:
        logging.info("Estimating Naive VAF from pooled reads...")
        p_germline.mle_vaf()
        logging.info("Finished VAF estimation from pooled reads!")
        logging.info("Starting EM-algorithm...")
        _, lambdas_hat, betas_hat, kappa_hat = p_germline.em_algo(
            algo=algo, delta_logll=delta
        )
        logging.info("Finished EM-algorithm!")
        logging.info("Estimating MLE VAF ...")
        ci_mle_p = p_germline.est_vaf_CI()
        logging.info("Finished estimating MLE VAF!")
        logging.info("Estimating posterior probability of germline heterozygosity...")
        pp_germline_poly = p_germline.post_prob_poly(
            lambdas=lambdas_hat, betas=betas_hat, kappa=kappa_hat
        )
    if lrt:
        logging.info("Estimating the naive likelihood ratio ...")
        loglik_ratio = p_germline.loglik_ratio_het(eps=eps)
        logging.info("Finished estimation of VAF and likelihood ratio!")
    if mutect2:
        logging.info("Estimating LOD Score under the Mutect2 Model ...")
        mutect_lod = MutectLOD(X=clone_reads)
        mutect_lod.lod_scores()
        mutect_lod.lod_germline()
        logging.info("Estimated LOD under Mutect2 model!")
    if betabinomial:
        logging.info("Estimating rho for Beta-Binomial overdispersion ...")
        beta_disp = BetaOverdispersion(X=clone_reads)
        rhos = beta_disp.estimate_rhos()
        logging.info("Estimated rho for Beta-Binomial overdispersion!")
    logging.info(f"Writing annotated VCF output to {out} ...")
    out_vcf = VCF(vcf, samples=samples, threads=nthreads)
    if em:
        out_vcf.add_info_to_header(
            {
                "ID": "ppGermlinePoly",
                "Number": 1,
                "Type": "Float",
                "Description": "Log posterior probability of germline polymorphism.",
            }
        )
        out_vcf.add_info_to_header(
            {
                "ID": "mleVAF",
                "Number": 1,
                "Type": "String",
                "Description": "MLE estimate of variant allele frequency and 95% CI.",
            }
        )
    if lrt:
        out_vcf.add_info_to_header(
            {
                "ID": "lrtGermlinePoly",
                "Number": 1,
                "Type": "Float",
                "Description": "likelihood ratio estimate of difference from germline heterozygote.",
            }
        )
    if mutect2:
        out_vcf.add_info_to_header(
            {
                "ID": "lodMutect",
                "Number": 1,
                "Type": "Float",
                "Description": "LOD Score from Mutect.",
            }
        )
    if betabinomial:
        out_vcf.add_info_to_header(
            {
                "ID": "rhobeta",
                "Number": 1,
                "Type": "Float",
                "Description": "Beta-Binomial overdispersion from Spencer-Chapman et al.",
            }
        )
    if em:
        if "germline" in config:
            for a, lhat in zip(["germline"] + config["annotations"], lambdas_hat):
                out_vcf.add_to_header(f"##lambda_{a}={lhat}")
        else:
            for a, lhat in zip(config["annotations"], lambdas_hat):
                out_vcf.add_to_header(f"##lambda_{a}={lhat}")
        out_vcf.add_to_header(f"##kappa_hat={kappa_hat}")
    out_vcf.add_to_header(f"##pGermlinePoly=run {' '.join(sys.argv[1:])}")
    write_vcf = Writer(fname=out, tmpl=out_vcf)
    write_vcf.write_header()
    i = 0
    for v in tqdm(out_vcf):
        if em:
            v.INFO["ppGermlinePoly"] = pp_germline_poly[i]
            v.INFO["mleVAF"] = f"{ci_mle_p[i, 0]}:{ci_mle_p[i, 1]}:{ci_mle_p[i, 2]}"
        if lrt:
            v.INFO["lrtGermlinePoly"] = loglik_ratio[i]
        if mutect2:
            v.INFO["lodMutect"] = mutect_lod.lod_germline[i]
        if betabinomial:
            v.INFO["rhobeta"] = rhos[i]
        write_vcf.write_record(v)
        i += 1
    write_vcf.close()
    logging.info(f"Wrote annotated VCF output to {out}!")
