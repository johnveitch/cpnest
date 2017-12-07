.. CPNest documentation master file, created by
   sphinx-quickstart on Thu Dec  7 13:57:03 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Welcome to CPNest's documentation!
==================================

Introduction
============

CPNest is a python package for performing Bayesian inference using the nested sampling algorithm.
It is designed to be simple for the user to provide a model via a set of parameters, their
bounds and a log-likelihood function. An optional log-prior function can be given
for non-uniform prior distributions.

The nested sampling algorithm is then used to compute
the marginal likelihood or evidence, $$ Z = \\int L ~dX. $$

As a by-product the algorithm produces samples from the posterior probability distribution.

The implementation is based on an ensemble MCMC sampler which can use multiple cores
to parallelise computation. It is compatible with both python 2.7+ and 3.5+.

Installation
============

The CPNest module is Free Software under the MIT license, and available on `Github <https://github.com/johnveitch/cpnest>`_ The simplest way to install cpnest is via pip::

  pip install cpnest

This is usually the best way to install the program. Alternatively, to install this package from source using setuptools::

  python setup.py install

Tests can be run with::

  python setup.py test

This documentation can be built in build/sphinx with::

  python setup.py build_sphinx -b html



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
