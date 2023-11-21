"""Setup module for building pGermlinePoly utilities"""


from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [Extension("poly_utils", ["pGermlinePoly/poly_utils.pyx"])]

setup_args = dict(ext_modules=cythonize(extensions))
setup(**setup_args)
