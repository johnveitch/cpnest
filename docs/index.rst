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
to parallelise computation. It is compatible with python 3.5+.

Installation
============

The CPNest module is Free Software under the MIT license, and available on `Github <https://github.com/johnveitch/cpnest>`_ The simplest way to install cpnest is via pip::

  pip install cpnest
  
If you are using conda it is possible to install from conda-forge::

  conda install -c conda-forge cpnest

This is usually the best way to install the program. Alternatively, to install this package from source using setuptools::

  git clone https://github.com/johnveitch/cpnest.git
  cd cpnest
  python3 setup.py install

Tests can be run with::

  python3 setup.py test

This documentation can be built in build/sphinx with::

  python3 setup.py build_sphinx -b html

Quickstart
==========

CPNest provides a nested sampling class that interfaces with a user-defined model,
which must implement the interface defined in :class:`cpnest.model.Model`.
The simplest way to do this is for the user to inherit from this class, and implement
the :func:`cpnest.model.Model.log_likelihood` function, and define the :obj:`cpnest.model.Model.names`
and :obj:`cpnest.model.Model.bounds` for their model. Here is an example for a two-dimensional
Gaussian distribution on variable x and y.

.. code-block:: python

        import numpy as np
        import cpnest

        class Simple2DModel(cpnest.model.Model):
            """
            A simple 2 dimensional gaussian
            """
            names=['x','y']
            bounds=[[-10,10],[-10,10]]

            def log_likelihood(self, param):
                return -0.5*(param['x']**2 + param['y']**2) - np.log(2.0*np.pi)

The user then uses CPNest by passing this when creating a :class:`cpnest.cpnest.CPNest`
object, which provides a way of controlling the parameters of the nested sampling run. The use then calls :func:`cpnest.cpnest.CPNest.run()`, which starts the run going:

.. code-block:: python

        mymodel = Simple2DModel()
        nest = cpnest.CPNest(mymodel)
        nest.run()

After calling `run()`, the final evidence and information will be displayed on the command line output:

.. code-block:: python

        >>> Final evidence: -5.99
        >>> Information: 3.37

Note that the final result will have some uncertainty that can be reduced by increasing the number of live points
with the `nlive` keyword argument.

The other keyword arguments for :obj:`cpnest.cpnest.CPNest`
provide means of controlling the number of threads used, the verbosity of the output, setting the random seed,
and so on. See the documentation for :obj:`cpnest.cpnest.CPNest` for more details. 

Retrieving output
==================

The log-evidence from the run is retrieved from :obj:`cpnest.cpnest.CPNest.NS.logZ`

The user can retrieve the samples produced during the run, and samples from the posterior
by calling the :func:`cpnest.cpnest.CPNest.get_nested_samples()` and :func:`cpnest.cpnest.CPNest.get_posterior_samples()`
methods, which both return a numpy array.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
