.. image:: _static/pGermlinePoly_logo.png
   :alt: pGermlinePoly logo
   :align: center
   :width: 80%

Welcome to pGermlinePoly's documentation!
==========================================

``pGermlinePoly`` is a Bayesian model to estimate the posterior probability of
germline polymorphism in somatic sequencing data. Annotation weights — capturing
how features such as population allele frequency or sequencing depth inform the
germline prior — are learned directly from the data via empirical Bayes rather
than specified by the user. The underlying EM algorithm jointly estimates these
logistic annotation weights and a Beta-Binomial error concentration parameter,
enabling data-driven discrimination between germline heterozygotes and somatic
variants.

Beyond the primary EM-based classifier, ``pGermlinePoly`` also provides
general-purpose tools for somatic variant filtering: a frequentist likelihood
ratio test (``--lrt``), a Mutect2 LOD score (``--mutect2``), and a
Beta-Binomial overdispersion statistic (``--betabinomial``). The tool is
designed to annotate somatic VCFs in-place — all scores are written directly
to the INFO fields of the input VCF — making it straightforward to integrate
into existing somatic variant calling pipelines.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   pGermlinePoly


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
