from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules=[
             Extension("cumulative",
                       sources=["cumulative.pyx"],
                       libraries=["m"], # Unix-like specific
                       include_dirs=[numpy.get_include()]
                       )
             ]

setup(
      name = "Cumulative",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
      )

ext_modules=[
             Extension("utils",
                       sources=["utils.pyx"],
                       libraries=["m"], # Unix-like specific
                       include_dirs=[numpy.get_include()]
                       )
             ]

setup(
      name = "utils",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
      )
