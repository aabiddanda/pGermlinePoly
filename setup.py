"""Setup module for building pGermlinePoly utilities."""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

if sys.platform == "darwin":
    omp_prefix = "/opt/homebrew/opt/libomp"
    extra_compile_args = ["-Xpreprocessor", "-fopenmp", f"-I{omp_prefix}/include"]
    extra_link_args = [f"-L{omp_prefix}/lib", "-lomp"]
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        "poly_utils",
        ["pGermlinePoly/poly_utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup_args = dict(
    ext_modules=cythonize(
        extensions, compiler_directives={"language_level": 3, "profile": False}
    )
)
setup(**setup_args)
