"""Python package to annotate somatic sequencing VCF files to estimate the probability of germline polymorphism.

pGermlinePoly implements an EM-algorithm incoporating site-level annotations .

Modules exported are:

* ProbGermline: class for estimating the posterior probability of germline polymorphism.
* ClonalSim: class for simulating clonal sequencing datasets
"""

__version__ = "0.0.1c"

from .io import *  # noqa
from .pGermlinePoly import ClonalSim, ProbGermline
