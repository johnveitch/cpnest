import unittest
import numpy as np
import cpnest.model

"""
Implements the Shrinkage test from Johannes Buchner
https://arxiv.org/abs/1407.5459

Tests the sampling of the shrinkage volume by sampling
a hyper-pyramid likelihood function that has a known relationship
between logL and logX
"""

class HyperPyramid(cpnest.model.Model):
    """
    HyperPyramid distribution
    """
    def __init__(self, dim=2, slope=100, sigma=1,**kwargs):
        self.dim = dim
        self.slope=slope
        self.names = ['x{0}'.format(i) for i in range(self.dim)]
        self.r0=1
        self.bounds = [(0,self.r0) for _ in self.names]
    def log_likelihood(self,x):
        sup = max(np.abs(y-0.5)/self.sigma for y in x)
        return -sup**(1.0/self.slope)
    
class ShrinkageTestCase(unittest.TestCase):
    def setUp(self):
        self.work=cpnest.CPNest( HyperPyramid(),
                                verbose=1,
                                Nthreads=4,
                                Nlive=1000,
                                maxmcmc=5000
                                )
    def test_run(self):
        self.work.run()
        self.assertTrue(1)

