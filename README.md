[![CI](https://github.com/aabiddanda/pGermlinePoly/actions/workflows/python-package.yml/badge.svg)](https://github.com/aabiddanda/pGermlinePoly/actions/workflows/python-package.yml)
[![Coverage](https://codecov.io/gh/aabiddanda/pGermlinePoly/graph/badge.svg)](https://codecov.io/gh/aabiddanda/pGermlinePoly)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://aabiddanda.github.io/pGermlinePoly/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<p align="center">
  <img src="docs/_static/pGermlinePoly_logo.png" alt="pGermlinePoly logo" width="70%">
</p>

# pGermlinePoly

`pGermlinePoly` is a Bayesian model to estimate the posterior probability of germline polymorphism in somatic sequencing data.

The model is flexible to account for a number of annotations that can be informative of distinguishing a true germline variant from an underlying somatic variant (e.g. global allele frequency). The underlying EM-algorithm estimates weights of how each annotation contributes to the probability of being a true germline polymorphism. After maximum-likelihood estimation of these parameters, the posterior probability of each variant is obtained.

## Installation

Currently the package is installable via `pip` using the following command:

```
pip install git+https://github.com/aabiddanda/pGermlinePoly.git
```

We highly recommend installing `pGermlinePoly` within a standalone environment (e.g., using [`uv`](https://docs.astral.sh/uv/)) Following this installation, you should be able to run either `pytest` to run the core unit tests for all of the functions in the library as well as testing out the two commandline executables detailed below.

## Running `pGermlinePoly`

The core algorithm for annotating a clonal sequencing VCF file is wrapped in the executable `pGermlinePoly`. At least one inference mode flag (`--em`, `--lrt`, `--mutect2`, or `--betabinomial`) must be specified.

```
Usage: pGermlinePoly [OPTIONS]

  Inference of annotation-informed probability of germline polymorphism from
  somatic clonal sequencing data using an EM algorithm that jointly estimates
  logistic annotation weights (lambda) and a Beta-Binomial error concentration
  (kappa).

Options:
  -v, --vcf PATH                  Input VCF file for all clones.  [required]
  -g, --germline_vcf PATH         Input VCF for germline sample.
  -c, --config PATH               Input config file detailing clonal
                                  structure.  [required]
  -t, --nthreads INTEGER          Number of threads.  [default: 1]
  -a, --algo [L-BFGS-B|Powell|Nelder-Mead]
                                  Optimization algorithm for the EM M-step.
                                  [default: L-BFGS-B]
  -e, --eps FLOAT                 Error rate for read-level alleles.
                                  [default: 0.001]
  -d, --delta FLOAT               EM convergence threshold (absolute change in
                                  log-likelihood).  [default: 0.0001]
  --em                            Run the EM algorithm to estimate annotation
                                  weights (lambda) and Beta-Binomial
                                  concentration (kappa).
  --lrt                           Frequentist likelihood ratio test testing
                                  deviation from germline heterozygote.
  --mutect2                       LOD Score from Mutect2.
  --betabinomial                  Implement the Beta-Binomial overdispersion
                                  method from Spencer-Chapman et al.
  -o, --out TEXT                  Output VCF file (defaults to stdout)
                                  [default: - required]
  --help                          Show this message and exit.
```


### Configuration Structure

The configuration yaml file for running `pGermlinePoly` has the following fields:

- `ind`: the overall identifier for the individual (not used for inference)
- `sex`: the sex of the individual (not used for inference)
- `age`: the age of the individual (not used for inference)
- `annotations`: the INFO field annotations used when constructing the EM-algorithm for weights. Each entry is either a plain string (INFO field name) or a dict with the following keys:
  - `field` *(required)*: the INFO field name to extract from the VCF.
  - `transform` *(optional)*: element-wise transform applied after extraction. Supported values: `"log10"`, `"sqrt"`.
  - `is_af` *(optional, boolean)*: set to `true` if this annotation is a population allele frequency (AF). AF annotations are automatically reflected (AF → 1−AF) for sites where the allele is reoriented to the minor allele, so the annotation continues to describe the minor allele.
- `clones`: the list of clone sample IDs — must match sample names in the clone VCF.
- `germline` *(optional)*: the germline sample IDs. If present, a separate germline VCF must be provided via `--germline_vcf`.

An example configuration is below:

```yaml
ind: IndA
sex: M
age: 50.0
annotations:
  - field: ExternalAF
    transform: log10
    is_af: true
  - DP
clones:
  - Aclone0
  - Aclone1
  - Aclone2
  - Aclone3
  - Aclone4
  - Aclone5
  - Aclone6
germline:
  - Agermline
```

### EM Algorithm (`--em`)

The primary inference mode. Runs an EM algorithm to jointly estimate logistic annotation weights (lambda) and the Beta-Binomial concentration parameter (kappa). After convergence, each site is annotated with:

- `ppGermlinePoly`: log posterior probability of germline polymorphism.
- `mleVAF`: MLE estimate of variant allele frequency with 95% CI (formatted as `mle:lower:upper`).

Estimated parameters (`lambda` per annotation and `kappa`) are written to the VCF header.

### Likelihood Ratio Test (`--lrt`)

A frequentist likelihood ratio test for site-level detection of somatic mutations, where the null hypothesis for a human germline heterozygote is $H_0: VAF = 0.5$ (assuming unbiased sampling). The per-site likelihood under the MLE VAF is compared to the null. Note that this null assumes diploid heterozygotes and may be inaccurate in regions with germline copy number changes.

P-values can be constructed under a chi-squared distribution. Annotates the output VCF with `lrtGermlinePoly`.

### Mutect2 LOD Score (`--mutect2`)

Estimates the LOD score for somatic variants using the Mutect2 model. Annotates the output VCF with `lodMutect`.

### Beta-Binomial Overdispersion (`--betabinomial`)

Implements the Beta-Binomial overdispersion approach from Spencer-Chapman et al. to detect read-count overdispersion indicative of somatic variants. Annotates the output VCF with `rhobeta`.


## Simulating somatic clonal sequencing data

A main feature of this package is that we actually are able to mimic the simulation of clonal sequencing data, where a germline sample is generated and an arbitrary number of clones are generated at varying levels of sequencing coverage for experimental validation. In the API, this is all held within the `ClonalSim` class. After installation, we also provide an easy to use simulation engine (`somatic-sim`) that generates VCF files as a default output that can then be used for subsequent inference. True somatic mutations will be indicated using the `SM` flag.


The generated VCF file will not be sorted by position by default (which can mess up indexing). The way around this is to directly pass the output to `bcftools sort ` and indexing:

```
somatic-sim [options] -o /dev/stdout | bcftools sort | bgzip > out.vcf.gz tabix -f out.vcf.gz
```

We encourage `tabix` indexing as a broader statement of the validity of the VCF file and that it will comply with other common tools.

The full suite of options for simulation are:

```
Usage: somatic-sim [OPTIONS]

  Simulation engine for neutral clonal sequencing data with a germline sample.

Options:
  -l, --seqlen INTEGER           Length of contig to simulate somatic
                                 mutations along.  [default: 10000000
                                 required]
  -j, --nclones INTEGER          Number of sampled clones.  [default: 5
                                 required]
  -a, --age FLOAT                Age of individual from clonal sampling.
                                 [default: 30.0 required]
  --afs_alpha FLOAT              Estimate of alpha parameter for allele
                                 frequency - default from NFE AF in gnomAD v3
                                 [default: 0.31699444395046117]
  --afs_beta FLOAT               Estimate of beta parameter for allele
                                 frequency - default from NFE AF in gnomAD v3
                                 [default: 6.067159920986527]
  -het, --germline_het FLOAT     Heterozygosity rate (per bp)  [default:
                                 0.001]
  -mu, --germline_mu FLOAT       Germline mutation rate (per bp / per
                                 generation)  [default: 1.2e-08]
  -smu, --somatic_mu FLOAT       Somatic mutation rate (per bp / per year)
                                 [default: 1.2e-09]
  -g, --mean_germline_cov FLOAT  Mean coverage in germline sample.  [default:
                                 15.0]
  -v, --sd_germline_cov FLOAT    Standard deviation in germline coverage.
                                 [default: 5.0]
  -gq, --germline_q FLOAT        Assumed read-quality score for germline.
                                 [default: 30.0]
  -gc, --mean_clone_cov FLOAT    Mean coverage in clone sequencing depth.
                                 [default: 15.0]
  -vc, --sd_clone_cov FLOAT      Standard deviation in clone sequencing depth.
                                 [default: 5.0]
  -cq, --clone_q FLOAT           Assumed read-quality score for clones.
                                 [default: 30.0]
  --seed INTEGER                 Random number seed for simulations.
                                 [default: 42]
  -o, --out TEXT                 Output VCF file  [default: out.vcf required]
  -ot, --out_tree TEXT           Output newick file for somatic clone
                                 genealogy (unscaled).
  -oc, --out_config TEXT         Output yaml-based config file for applying
                                 inference post-hoc.
  --help                         Show this message and exit.

```
