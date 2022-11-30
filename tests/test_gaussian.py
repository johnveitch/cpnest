import unittest
import numpy as np
import cpnest.model

class GaussianModel(cpnest.model.Model):
    """
    A simple gaussian model with parameters mean and sigma
    """
    names=['mean','sigma']
    bounds=[[-5,5],[0.05,1]]
    data = np.array([x for x in np.random.normal(0.5,0.5,size=10)])
    analyticZ = np.log(0.05)

    @classmethod
    def log_likelihood(cls,x):
        return -0.5*x['mean']**2/x['sigma']**2 - np.log(x['sigma']) - 0.5*np.log(2.0*np.pi)

    def log_prior(self,p):
        if not self.in_bounds(p): return -np.inf
        return -np.log(p['sigma']) - np.log(10) - np.log(0.95)

    def force(self,x):
        return np.zeros(1, dtype = {'names':x.names, 'formats':['f8' for _ in x.names]})

class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.model = GaussianModel()
        self.work=cpnest.CPNest(self.model,verbose=0,nthreads=8,nlive=100,maxmcmc=100)

    def test_run(self):
        self.work.run()
        print('Analytic evidence: {0}'.format(self.model.analyticZ))



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    freeze_support()
    unittest.main(verbosity=2)

