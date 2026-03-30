"""CLI for pGermlinePoly Estimation in VCF."""

import logging

import rich_click as click
from cyvcf2 import VCF, Writer
from tqdm import tqdm

from pGermlinePoly import ProbGermline
from pGermlinePoly.io import (
    validate_config,
    check_samples,
    check_annotations,
    create_germline_anno_gl,
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
    help="Input config file detailing clone structure.",
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
    "--vaf",
    required=False,
    default=False,
    is_flag=True,
    type=bool,
    help="Estimate the variant allele frequency.",
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
def main(vcf, config, nthreads, algo, naive, eps, lrt, vaf, mutect2, betabinomial, out):
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
    germline_anno = create_germline_anno_gl(germline_vcf)
    clonal_vcf = VCF(vcf, samples=config["clones"], threads=nthreads)
    clone_pl = create_clonal_pl_matrix(clonal_vcf)
    anno_vcf = VCF(vcf, samples=config["clones"], threads=nthreads)
    anno = create_anno(anno_vcf, annotations=annotations)
    # this is a little strange in terms of dimensions here ...
    full_anno = np.vstack([germline_anno, anno]).T
    logging.info("Finished extracting data for inference!")

    p_germline = ProbGermline(X=clone_pl, Theta=full_anno)
    logging.info("Imputing missing annotations...")
    p_germline.impute_anno()
    if naive:
        logging.info("Starting Numerical MLE estimation!")
        a0_hat, lambdas_hat = p_germline.naive_mle(algo=algo, disp=(out != "-"))
        logging.info("Finished Numerical MLE estimation!")

    else:
        logging.info("Starting EM-algorithm...")
        loglls, a0_hats, lambdas_hats = p_germline.em_algo(
            algo=algo,
            delta_logll=eps,
        )
        a0_hat = a0_hats[-1]
        lambdas_hat = lambda_hats[-1]
        logging.info("Finished EM-algorithm!")
    logging.info("Estimating posterior probability of germline heterozygosity...")
    pp_germline_poly = p_germline.post_prob_poly(lambdas=lambdas_hat, a0=a0_hat)
    if lrt or vaf:
        logging.info("Estimating of VAF and likelihood ratio ...")
        mle_p, logll_p, ci_mle_p = p_germline.est_vaf_CI()
        loglik_ratio = p_germline.loglik_ratio(logll_p=logll_p)
        logging.info("Finished estimation of VAF and likelihood ratio!")
    if mutect2:
        logging.info("Estimating LOD Score under the Mutect2 Model ...")
        # TODO: do something here ...
        logging.info("Estimated LOD under Mutect2 model!")
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
    if lrt or vaf:
        out_vcf.add_info_to_header(
            {
                "ID": "mleVAF",
                "Number": 1,
                "Type": "String",
                "Description": "MLE estimate of variant allele frequency and 95% CI.",
            }
        )
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
    for a, l in zip(["germline"] + config["annotations"], lambdas_hat):
        out_vcf.add_to_header(f"##lambda_{a}={l}")
    write_vcf = Writer(fname=out, tmpl=out_vcf)
    write_vcf.write_header()
    if lrt or vaf:
        for pp_gp, lrt, vaf, vaf_low, vaf_high, v in tqdm(
            zip(
                pp_germline_poly,
                loglik_ratio,
                mle_p,
                ci_mle_p[:, 0],
                ci_mle_p[:, 2],
                out_vcf,
            )
        ):
            v.INFO["ppGermlinePoly"] = pp_gp
            v.INFO["mleVAF"] = f"{max(vaf_low, 0.0)}:{vaf}:{min(vaf_high, 1.0)}"
            v.INFO["lrtGermlinePoly"] = lrt
            write_vcf.write_record(v)
    else:
        for pp_gp, v in tqdm(zip(pp_germline_poly, out_vcf)):
            v.INFO["ppGermlinePoly"] = pp_gp
            write_vcf.write_record(v)
    write_vcf.close()
    logging.info(f"Wrote annotated VCF output to {out}!")
