"""Python package to annotate somatic sequencing VCF files to estimate the probability of germline polymorphism.

pGermlinePoly implements an EM-algorithm incoporating site-level annotations .

Modules exported are:

* ClonalSim: class for simulating clonal sequencing datasets.
* MutectLOD: class for estimating LOD score for somatic variants using the Mutect2 model.
* ProbGermline: class for estimating the posterior probability of germline polymorphism.
* BetaOverdispersion: class for estimating overdispersion indicative of somatic variants using Beta-binomial model.
"""

__version__ = "0.0.2a"

from .io import *  # noqa
from .pGermlinePoly import ClonalSim, MutectLOD, ProbGermline, BetaOverdispersion  # noqa
