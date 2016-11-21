from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules=[
             Extension("parameter",
                       sources=["parameter.pyx"],
                       libraries=["m","gsl","gslcblas"], # Unix-like specific
                       include_dirs=[numpy.get_include()]
                       )
             ]

setup(
      name = "parameter",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
      )

ext_modules=[
             Extension("proposals",
                       sources=["proposals.pyx"],
                       libraries=["m"], # Unix-like specific
                       include_dirs=[numpy.get_include()]
                       )
             ]

setup(
      name = "proposals",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
      )

ext_modules=[
             Extension("NestedSampling",
                       sources=["NestedSampling.pyx"],
                       libraries=["m"], # Unix-like specific
                       include_dirs=[numpy.get_include()]
                       )
             ]

setup(
      name = "NestedSampling",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
      )

ext_modules=[
             Extension("cosmology",
                       sources=["cosmology.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = ["/Users/wdp/opt/lalsuite/lib"],
                       include_dirs=[numpy.get_include(),"/Users/wdp/opt/lalsuite/include"]
                       )
             ]

setup(
      name = "cosmology",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include(),"/Users/wdp/opt/lalsuite/include"]
      )