"""CLI for pGermlinePoly Estimation in VCF."""
import logging
import sys

import click
from cyvcf2 import VCF, Writer
from tqdm import tqdm

from pGermlinePoly import ProbGermline
from pGermlinePoly.io import *  # noqa

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
    "--out",
    "-o",
    required=True,
    type=str,
    default="-",
    help="Output VCF file (defaults to stdout)",
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
    p_germline.impute_anno()
    loglls, lambdas_hat = p_germline.em_algo(
        lambdas=np.zeros(p_germline.A, dtype="double")
    )
    logging.info("Finished EM-algorithm!")
    logging.info("Estimating posterior probability of germline heterozygosity...")
    pp_germline_poly = p_germline.post_prob_poly(lambdas=lambdas_hat)
    logging.info(f"Writing VCF output to {out} w/ ppGermlinePoly...")
    out_vcf = VCF(vcf, samples=samples, threads=nthreads)
    out_vcf.add_info_to_header(
        {
            "ID": "ppGermlinePoly",
            "Number": 1,
            "Type": "Float",
            "Description": "Posterior probability of germline polymorphism.",
        }
    )
    write_vcf = Writer(fname=out, tmpl=out_vcf)
    write_vcf.write_header()
    for pp_gp, v in tqdm(zip(pp_germline_poly, out_vcf)):
        v.INFO["ppGermlinePoly"] = pp_gp
        write_vcf.write_record(v)
    write_vcf.close()
    logging.info(f"Wrote annotated VCF output to {out}!")
