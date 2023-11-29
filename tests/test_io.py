import pytest
import yaml
from cyvcf2 import VCF

from pGermlinePoly.io import check_annotations, check_samples, validate_config

# -------- 1. Testing Configs --------- #
good_config = """
ind: JH214UJ
age: 50
sex: M
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations:
  - MLEAF
"""

no_anno_config = """
ind: JH214UJ
age: 50
sex: M
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations: []
"""

bad_sex_config = """
ind: JH214UJ
age: 50
sex: X
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations:
  - MLEAF
"""

no_sex_config = """
ind: JH214UJ
age: 50
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations:
  - MLEAF
"""

bad_age_config = """
ind: JH214UJ
age: -1
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations:
  - MLEAF
"""

no_age_config = """
ind: JH214UJ
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations:
  - MLEAF
"""


def test_valid_config_strings(tmp_path):
    """Testing valid configurations."""
    d = tmp_path / "valid"
    d.mkdir()
    p1 = d / "hello.yaml"
    p1.write_text(good_config)
    validate_config(p1)
    p2 = d / "hello2.yaml"
    p2.write_text(no_anno_config)
    validate_config(p2)


def test_bad_config_strings(tmp_path):
    """Testing bad configs in various cases."""
    d = tmp_path / "invalid"
    d.mkdir()
    for c_str in [bad_sex_config, no_sex_config, bad_age_config, no_age_config]:
        p1 = d / "hello.yaml"
        p1.write_text(c_str)
        with pytest.raises(Exception):
            validate_config(p1)


# --------- 2. Testing VCFs ---------- #
good_vcf_string = """
##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##ALT=<ID=NON_REF,Description="Represents any possible alternative allele not already represented at this location by REF and ALT">
##FILTER=<ID=LowQual,Description="Low quality">
##FILTER=<ID=VQSRTrancheINDEL99.90to99.95,Description="Truth sensitivity tranche level for INDEL model at VQS Lod: -26.1487 <= x < -16.8809">
##FILTER=<ID=VQSRTrancheINDEL99.95to100.00+,Description="Truth sensitivity tranche level for INDEL model at VQS Lod < -2854.9916">
##FILTER=<ID=VQSRTrancheINDEL99.95to100.00,Description="Truth sensitivity tranche level for INDEL model at VQS Lod: -2854.9916 <= x < -26.1487">
##FILTER=<ID=VQSRTrancheSNP99.80to99.90,Description="Truth sensitivity tranche level for SNP model at VQS Lod: -3.793 <= x < -2.5623">
##FILTER=<ID=VQSRTrancheSNP99.90to99.95,Description="Truth sensitivity tranche level for SNP model at VQS Lod: -5.969 <= x < -3.793">
##FILTER=<ID=VQSRTrancheSNP99.95to100.00+,Description="Truth sensitivity tranche level for SNP model at VQS Lod < -26560.1627">
##FILTER=<ID=VQSRTrancheSNP99.95to100.00,Description="Truth sensitivity tranche level for SNP model at VQS Lod: -26560.1627 <= x < -5.969">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth (reads with MQ=255 or with bad mates are filtered)">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=MIN_DP,Number=1,Type=Integer,Description="Minimum DP observed within the GVCF block">
##FORMAT=<ID=PGT,Number=1,Type=String,Description="Physical phasing haplotype information, describing how the alternate alleles are phased in relation to one another; will always be heterozygous and is not intended to describe called alleles">
##FORMAT=<ID=PID,Number=1,Type=String,Description="Physical phasing ID information, where each unique ID within a given sample (but not across samples) connects records within a phasing group">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phasing set (typically the position of the first variant in the set)">
##FORMAT=<ID=RGQ,Number=1,Type=Integer,Description="Unconditional reference genotype confidence, encoded as a phred quality -10*log10 p(genotype call is wrong)">
##FORMAT=<ID=SB,Number=4,Type=Integer,Description="Per-sample component statistics which comprise the Fisher's Exact Test to detect strand bias.">
##GATKCommandLine=<ID=ApplyVQSR,CommandLine="ApplyVQSR --recal-file /scratch16/rmccoy22/Armanios_Collab/Short_Telomeres/Mapped_libraries/Preprocessed_alignments/HaplotypeCaller/Genotyped_VCFs/ShortTLProject_chr11.sites_only.indels.recal --tranches-file ShortTLProject_chr11.sites_only.indels.tranches --output ShortTLProject_chr11.indel.recalibrated.vcf.gz --truth-sensitivity-filter-level 99.7 --mode INDEL --variant ShortTLProject_chr11.trimmed_samples.vcf.gz --create-output-variant-index true --use-allele-specific-annotations false --ignore-all-filters false --exclude-filtered false --interval-set-rule UNION --interval-padding 0 --interval-exclusion-padding 0 --interval-merging-rule ALL --read-validation-stringency SILENT --seconds-between-progress-updates 10.0 --disable-sequence-dictionary-validation false --create-output-bam-index true --create-output-bam-md5 false --create-output-variant-md5 false --max-variants-per-shard 0 --lenient false --add-output-sam-program-record true --add-output-vcf-command-line true --cloud-prefetch-buffer 40 --cloud-index-prefetch-buffer -1 --disable-bam-index-caching false --sites-only-vcf-output false --help false --version false --showHidden false --verbosity INFO --QUIET false --use-jdk-deflater false --use-jdk-inflater false --gcs-max-retries 20 --gcs-project-for-requester-pays  --disable-tool-default-read-filters false",Version="4.2.6.1",Date="October 31, 2023 10:17:24 AM EDT">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency, for each ALT allele, in the same order as listed">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
##INFO=<ID=BaseQRankSum,Number=1,Type=Float,Description="Z-score from Wilcoxon rank sum test of Alt Vs. Ref base qualities">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth; some reads may have been filtered">
##INFO=<ID=END,Number=1,Type=Integer,Description="Stop position of the interval">
##INFO=<ID=ExcessHet,Number=1,Type=Float,Description="Phred-scaled p-value for exact test of excess heterozygosity">
##INFO=<ID=FS,Number=1,Type=Float,Description="Phred-scaled p-value using Fisher's exact test to detect strand bias">
##INFO=<ID=InbreedingCoeff,Number=1,Type=Float,Description="Inbreeding coefficient as estimated from the genotype likelihoods per-sample when compared against the Hardy-Weinberg expectation">
##INFO=<ID=MLEAC,Number=A,Type=Integer,Description="Maximum likelihood expectation (MLE) for the allele counts (not necessarily the same as the AC), for each ALT allele, in the same order as listed">
##INFO=<ID=MLEAF,Number=A,Type=Float,Description="Maximum likelihood expectation (MLE) for the allele frequency (not necessarily the same as the AF), for each ALT allele, in the same order as listed">
##INFO=<ID=MQ,Number=1,Type=Float,Description="RMS Mapping Quality">
##INFO=<ID=MQRankSum,Number=1,Type=Float,Description="Z-score From Wilcoxon rank sum test of Alt vs. Ref read mapping qualities">
##INFO=<ID=NEGATIVE_TRAIN_SITE,Number=0,Type=Flag,Description="This variant was used to build the negative training set of bad variants">
##INFO=<ID=POSITIVE_TRAIN_SITE,Number=0,Type=Flag,Description="This variant was used to build the positive training set of good variants">
##INFO=<ID=QD,Number=1,Type=Float,Description="Variant Confidence/Quality by Depth">
##INFO=<ID=RAW_MQandDP,Number=2,Type=Integer,Description="Raw data (sum of squared MQ and total depth) for improved RMS Mapping Quality calculation. Incompatible with deprecated RAW_MQ formulation.">
##INFO=<ID=ReadPosRankSum,Number=1,Type=Float,Description="Z-score from Wilcoxon rank sum test of Alt vs. Ref read position bias">
##INFO=<ID=SOR,Number=1,Type=Float,Description="Symmetric Odds Ratio of 2x2 contingency table to detect strand bias">
##INFO=<ID=VQSLOD,Number=1,Type=Float,Description="Log odds of being a true variant versus being false under the trained gaussian mixture model">
##INFO=<ID=culprit,Number=1,Type=String,Description="The annotation which was the worst performing in the Gaussian mixture model, likely the reason why the variant was filtered out">
##bcftools_viewCommand=view --threads 4 -O z -o ShortTLProject_chr11.trimmed_samples.vcf.gz --samples-file ^bad_samples.txt ShortTLProject_chr11.vcf.gz; Date=Tue Oct 31 09:10:36 2023
##bcftools_viewVersion=1.17+htslib-1.17
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr2,length=242193529>
##contig=<ID=chr3,length=198295559>
##contig=<ID=chr4,length=190214555>
##contig=<ID=chr5,length=181538259>
##contig=<ID=chr6,length=170805979>
##contig=<ID=chr7,length=159345973>
##contig=<ID=chr8,length=145138636>
##contig=<ID=chr9,length=138394717>
##contig=<ID=chr10,length=133797422>
##contig=<ID=chr11,length=135086622>
##contig=<ID=chr12,length=133275309>
##contig=<ID=chr13,length=114364328>
##contig=<ID=chr14,length=107043718>
##contig=<ID=chr15,length=101991189>
##contig=<ID=chr16,length=90338345>
##contig=<ID=chr17,length=83257441>
##contig=<ID=chr18,length=80373285>
##contig=<ID=chr19,length=58617616>
##contig=<ID=chr20,length=64444167>
##contig=<ID=chr21,length=46709983>
##contig=<ID=chr22,length=50818468>
##contig=<ID=chrX,length=156040895>
##contig=<ID=chrY,length=57227415>
##contig=<ID=chrM,length=16569>
##source=ApplyVQSR
##bcftools_viewCommand=view -r chr21 --threads 8 -O z -o ShortTLProject.trimmed_samples.VQSR.chr21.vcf.gz ShortTLProject.trimmed_samples.VQSR.vcf.gz; Date=Tue Nov 14 12:14:42 2023
##bcftools_viewVersion=1.18+htslib-1.18
##bcftools_viewCommand=view -v snps -m2 -M2 --threads 4 ShortTLProject.trimmed_samples.VQSR.chr21.vcf.gz; Date=Wed Nov 22 22:19:49 2023
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	A-bulk	A-clone1	A-clone2
chr21	5030625	.	G	C	60.92	PASS	AC=0;AF=0.001773;AN=354;BaseQRankSum=0;DP=1411;ExcessHet=0;FS=0;InbreedingCoeff=0.4763;MLEAC=2;MLEAF=0.003546;MQ=48.76;MQRankSum=1.38;NEGATIVE_TRAIN_SITE;QD=15.23;ReadPosRankSum=0.674;SOR=0.693;VQSLOD=-1.88;culprit=MQ	GT:AD:DP:GQ:PGT:PID:PL:PS	0/0:0,0:0:0:.:.:0,0,0:.	0/0:11,0:11:33:.:.:0,33,433:.	0/0:8,0:8:24:.:.:0,24,247:.
"""

good_config_real = """
ind: A
age: 50
sex: M
germline:
  - A-Bulk
clones:
  - A-clone1
  - A-clone2
annotations: []
"""


def test_validate_vcf(tmp_path):
    """Run through a full validation of the VCF protocol."""
    d = tmp_path / "real_vcf_test"
    d.mkdir()
    config_fp = d / "config.yaml"
    config_fp.write_text(good_config_real)
    vcf_fp = d / "test.vcf"
    vcf_fp.write_text(good_vcf_string)
    config = validate_config(config_fp)
    samples = config["germline"] + config["clones"]
    annotations = config["annotations"]
    cur_vcf = VCF(vcf_fp, samples=samples)
    check_samples(cur_vcf, samples=samples)
    check_annotations(cur_vcf, annotations=annotations)
