"""CLI for pGermlinePoly Estimation in VCF."""

import logging

import rich_click as click
from cyvcf2 import VCF, Writer

from pGermlinePoly import ProbGermline, MutectLOD, BetaOverdispersion
from pGermlinePoly.io import (
    validate_config,
    check_samples,
    check_annotations,
    create_germline_anno,
)

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command(
    help="Inference of annotation-informed probability of germline polymorphism from somatic sequencing data.",
    context_settings=dict(show_default=True),
)
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
    help="Optimization algorithm for EM-algorithm or naive optimization.",
)
@click.option(
    "--naive",
    "-n",
    required=False,
    default=True,
    is_flag=True,
    type=bool,
    help="Numerical optimization of MLE parameters.",
)
@click.option(
    "--eps",
    "-e",
    required=False,
    default=1e-4,
    type=float,
    help="Stopping criteria for log-likelihood changes in the EM-algorithm",
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
def main(vcf, config, nthreads, algo, naive, eps, lrt, mutect2, betabinomial, out):
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
        # NOTE: this could be provided as an alternative VCF as well ...
        logging.info(f"Extracting germline annotation from {vcf} for inference...")
        germline_vcf = VCF(vcf, samples=config["germline"], threads=nthreads)
        germline_anno = create_germline_anno(germline_vcf)
        logging.info(f"Extracted germline annotation from {vcf} for inference...")
        full_anno = np.vstack([germline_anno, anno]).T
    else:
        full_anno = anno
    logging.info("Finished extracting clonal data for inference!")

    p_germline = ProbGermline(X=clone_reads, Theta=full_anno)
    logging.info("Imputing missing annotations...")
    p_germline.impute_anno()
    logging.info("Finished imputing missing annotations!")
    logging.info("Estimating Naive VAF from pooled reads...")
    p_germline.mle_vaf()
    logging.info("Finished VAF estimation from pooled reads!")
    if naive:
        logging.info("Starting Numerical MLE estimation!")
        lambdas_hat = p_germline.naive_mle(algo=algo, disp=(out != "-"))
        logging.info("Finished Numerical MLE estimation!")
    else:
        logging.info("Starting EM-algorithm...")
        loglls, lambdas_hats = p_germline.em_algo(
            algo=algo,
            delta_logll=eps,
        )
        lambdas_hat = lambda_hats[-1]
        logging.info("Finished EM-algorithm!")
    logging.info("Estimating posterior probability of germline heterozygosity...")
    pp_germline_poly = p_germline.post_prob_poly(lambdas=lambdas_hat)
    if lrt:
        logging.info("Estimating the naive likelihood ratio ...")
        mle_p, logll_p, ci_mle_p = p_germline.est_vaf_CI()
        loglik_ratio = p_germline.loglik_ratio(logll_p=logll_p)
        logging.info("Finished estimation of VAF and likelihood ratio!")
    if mutect2:
        logging.info("Estimating LOD Score under the Mutect2 Model ...")
        # TODO: do something here ...
        mutect_lod = MutectLOD(X=clone_reads)
        logging.info("Estimated LOD under Mutect2 model!")
    if betabinomial:
        logging.info("Estimating rho for Beta-Binomial overdispersion ...")
        beta_disp = BetaOverdispersion(X=clone_reads)
        rhos = beta_disp.estimate_rhos()
        logging.info("Estimated rho for Beta-Binomial overdispersion!")
    logging.info(f"Writing annotated VCF output to {out} ...")
    out_vcf = VCF(vcf, samples=samples, threads=nthreads)
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
    for a, l in zip(["germline"] + config["annotations"], lambdas_hat):
        out_vcf.add_to_header(f"##lambda_{a}={l}")
    write_vcf = Writer(fname=out, tmpl=out_vcf)
    write_vcf.write_header()
    # if lrt and mutect2:
    #     for pp_gp, lrt, vaf, vaf_low, vaf_high, v in tqdm(
    #         zip(
    #             pp_germline_poly,
    #             loglik_ratio,
    #             mle_p,
    #             ci_mle_p[:, 0],
    #             ci_mle_p[:, 2],
    #             out_vcf,
    #         )
    #     ):
    #         v.INFO["ppGermlinePoly"] = pp_gp
    #         v.INFO["mleVAF"] = f"{max(vaf_low, 0.0)}:{vaf}:{min(vaf_high, 1.0)}"
    #         v.INFO["lrtGermlinePoly"] = lrt
    #         write_vcf.write_record(v)
    # elif lrt:
    #     for pp_gp, lrt, vaf, vaf_low, vaf_high, v in tqdm(
    #         zip(
    #             pp_germline_poly,
    #             loglik_ratio,
    #             mle_p,
    #             ci_mle_p[:, 0],
    #             ci_mle_p[:, 2],
    #             out_vcf,
    #         )
    #     ):
    #         v.INFO["ppGermlinePoly"] = pp_gp
    #         v.INFO["mleVAF"] = f"{max(vaf_low, 0.0)}:{vaf}:{min(vaf_high, 1.0)}"
    #         v.INFO["lrtGermlinePoly"] = lrt
    #         write_vcf.write_record(v)
    # else:
    #     for pp_gp, v in tqdm(zip(pp_germline_poly, out_vcf)):
    #         v.INFO["ppGermlinePoly"] = pp_gp
    #         write_vcf.write_record(v)
    write_vcf.close()
    logging.info(f"Wrote annotated VCF output to {out}!")
