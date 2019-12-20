#!/bin/usr/env python

import unittest
import numpy as np
import cpnest.model


class RosenbrockModel(cpnest.model.Model):
    """
    The n-dimensional Rosenbrock function

    See: https://arxiv.org/pdf/1903.09556.pdf
    """
    def __init__(self, ndims=2):
        self.ndims = ndims
        self.names = ['x' + str(i) for i in range(ndims)]
        self.bounds = [[-5, 5] for i in range(ndims)]


    def log_likelihood(self, x):
        x = np.array(x.values).reshape(-1, self.ndims)
        return  -(np.sum(100. * (x[:,1:] - x[:,:-1] ** 2.) ** 2. + (1. - x[:,:-1]) ** 2., axis=1))

    def force(self, x):
        f = np.zeros(1, dtype = {'names':x.names, 'formats':['f8' for _ in x.names]})
        return f


class RosenbrockTestCase(unittest.TestCase):
    """
    Test the Rosenbrock model
    """
    def setUp(self):
        ndims = 2
        model = RosenbrockModel(ndims)
        self.work=cpnest.CPNest(model, verbose=2, nthreads=4, nlive=1000, maxmcmc=1000, poolsize=1000)

    def test_run(self):
        self.work.run()


if __name__=='__main__':
    unittest.main(verbosity=2)
