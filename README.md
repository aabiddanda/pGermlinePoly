# pGermlinePoly
Bayesian model to estimate posterior probability of germline polymorphism in somatic sequencing data.

The model is flexible to account for a number of annotations that can be informative of distinguishing a true germline variant from an underlying somatic variant (e.g. global allele frequency & germline genotype likelihood). The underlying EM-algorithm estimates weights of how each annotation contributes to the probability of being a true germline polymorphism. After maximum-likelihood estimation of these parameters, the posterior probability of each variant is obtained.


## Installation

Currently the package is installable via a local `pip install` using the following:

```
git clone git@github.com:aabiddanda/pGermlinePoly.git
cd pGermlinePoly pip install .
```

Following this installation, you should be able to run either `pytest` to run the core unit tests for all of the functions in the library as well as testing out the two commandline executables detailed below.

## Running `pGermlinePoly`

The core algorithm for annotating a clonal sequencing VCF file is wrapped in the executable `pGermlinePoly`.

```
Usage: pGermlinePoly [OPTIONS]

  Inference of annotation-informed probability of germline polymorphism from
  somatic sequencing data.

Options:
  -v, --vcf PATH                  Input VCF file.  [required]
  -c, --config PATH               Input config file detailing clone structure.
                                  [required]
  -t, --nthreads INTEGER          Number of threads.  [default: 1]
  -a, --algo [L-BFGS-B|Powell|Nelder-Mead]
                                  Optimization algorithm for EM-algorithm or
                                  naive optimization.  [default: L-BFGS-B]
  -n, --naive                     Numerical optimization of MLE parameters.
                                  [default: True]
  --lrt                           Frequentist likelihood ratio test testing
                                  deviation from germline heterozygote.
  --vaf                           Estimate the variant allele frequency.
  -o, --out TEXT                  Output VCF file (defaults to stdout)
                                  [default: - required]
  --help                          Show this message and exit.
```


### Configuration Structure

The configuration yaml file for running `pGermlinePoly`

The minimal fields have been:

- `ind`: the overall identifier for the individual (not used for inference)
- `sex`: the sex of the individual (not used for inference)
- `age`: the age of the individual (not used for inference)
- `annotations`: the annotations that are used when constructing the EM-algorithm for weights.
- `clones`: the list of clone IDs - that should also be in the VCF file.
- `germline`: the germline sample IDs.

An example configuration is below:

```
ind: IndA
sex: M
age: 50.0
annotations:
	- ExternalAF
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

### VAF Estimation

One major goal of clonal sequencing is to estimate the frequency of associated somatic variant, the ``variant allele fraction'' or VAF. If clear genotyping data is available then the maximum-likelihood estimator of the VAF is a good one, but uncertainty of this quantity is rarely reported.

We use a frequentist approach and use the inverse of the Fisher Information for each site as a way to quantify the 95% confidence interval of the VAF. To include the VAF confidence interval as an annotation in the resulting VCF, use the `--vaf` flag when running `pGermlinePoly`.

### Likelihood Ratio Test for Somatic Variants

We also have implemented a likelihood ratio test for site-level detection of somatic mutations, where the underlying null hypothesis for a human germline heterozygote is $H_0: VAF = 0.5$ (assuming that the sampling process is unbiased). We directly evaluate this using the per-site likelihood under the maximum-likelihood estimate of the VAF, and compare to the null hypothesis. Note that this null hypothesis assumes diploid and largely focuses on heterozygotes so may be inaccurate in regions where there is a germline deletion or duplication.

This can also be used effectively as a frequentist analog of the filtering criteria for a somatic mutation (which notably does not rely on annotations for priors), and p-values can also be constructed under a chi-squared distribution. In order to annotate the resulting VCF with the likelihood ratio test statistic use the `--lrt` flag.


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
  -v, --var_germline_cov FLOAT   Variance in germline coverage.  [default:
                                 5.0]
  -gq, --germline_q FLOAT        Assumed read-quality score for germline.
                                 [default: 30.0]
  -gc, --mean_clone_cov FLOAT    Mean coverage in clone sequencing depth.
                                 [default: 15.0]
  -vc, --var_clone_cov FLOAT     Variance in clone sequencing depth.
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
