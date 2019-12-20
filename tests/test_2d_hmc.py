import unittest
import numpy as np
import cpnest.model

class GaussianModel(cpnest.model.Model):
    """
    A simple 2 dimensional gaussian
    """
    def __init__(self):
        pass
    names=['x','y']
    bounds=[[-10,10],[-10,10]]
    analytic_log_Z=0.0 - np.log(bounds[0][1]-bounds[0][0]) - np.log(bounds[1][1]-bounds[1][0])

    def log_likelihood(self,p):
        return -0.5*(p['x']**2 + p['y']**2) - np.log(2.0*np.pi)

    def log_prior(self,p):
        return super(GaussianModel,self).log_prior(p)
    
    def force(self, p):
        f = np.zeros(1, dtype = {'names':p.names, 'formats':['f8' for _ in p.names]})
        return f

class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(GaussianModel(),verbose=1,nthreads=1,nlive=1000,maxmcmc=5000,poolsize=1000,nhamiltonian=1)

    def test_run(self):
        self.work.run()
        # 2 sigma tolerance
        tolerance = 2.0*np.sqrt(self.work.NS.state.info/self.work.NS.Nlive)
        self.assertTrue(np.abs(self.work.NS.logZ - GaussianModel.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0} +/- {2} instead of {1}'.format(self.work.NS.logZ ,GaussianModel.analytic_log_Z, tolerance))

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
 
