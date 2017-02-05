import unittest
import numpy as np
from scipy import integrate
import cpnest

class GaussianModel(cpnest.model.Model):
    """
    A simple gaussian model with parameters mean and sigma
    """
    def __init__(self):
        pass
    names=['x']
    bounds=[[-10,10]]
    analytic_log_Z=0.0 - np.log(bounds[0][1] - bounds[0][0])

    @classmethod
    def log_likelihood(cls,p):
        return -0.5*(p['x']**2) - 0.5*np.log(2.0*np.pi)


class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(GaussianModel(),verbose=2,Nthreads=1,Nlive=50,maxmcmc=1000)

    def test_run(self):
        self.work.run()
        # 2 sigma tolerance
        tolerance = 2.0*np.sqrt(self.work.NS.state.info/self.work.NS.Nlive)
        print('Tolerance: {0:0.3f}'.format(tolerance))
        self.assertTrue(np.abs(self.work.NS.logZ - GaussianModel.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0:.3f} instead of {1:.3f}'.format(self.work.NS.logZ,GaussianModel.analytic_log_Z ))



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
 
