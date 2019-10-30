#!/bin/usr/env python

import unittest
import numpy as np
import cpnest.model


class RosenbrockModel(cpnest.model.Model):
    """
    The two dimensional Rosenbrock density

    See: https://arxiv.org/pdf/1903.09556.pdf
    """
    def __init__(self, ndims=2):
        self.names = ['x1', 'x2']
        self.bounds = [[-5, 5], [-5, 5]]


    def log_likelihood(self, x):
        return - (100. * (x['x2'] - x['x1'] ** 2.) ** 2. + (1. - x['x1']) ** 2.)

    def force(self, x):
        f = np.zeros(1, dtype = {'names':x.names, 'formats':['f8' for _ in x.names]})
        return f

class RosenbrockTestCase(unittest.TestCase):
    """
    Test the Rosenbrock model
    """
    def setUp(self):
        self.model = RosenbrockModel()
        self.work = cpnest.CPNest(self.model, verbose=2, nthreads=4, nlive=1000, maxmcmc=1000, poolsize=1000)

    def test_run(self):
        self.work.run()

if __name__=='__main__':
    unittest.main(verbosity=2)
