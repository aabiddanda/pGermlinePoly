"""Setup module for building pGermlinePoly utilities."""

import numpy as np
from Cython.Build import cythonize
from distutils.core import Extension
from setuptools import setup

extensions = [
    Extension(
        "poly_utils",
        ["pGermlinePoly/poly_utils.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup_args = dict(
    ext_modules=cythonize(
        extensions, compiler_directives={"language_level": 3, "profile": False}
    )
)
setup(**setup_args)
