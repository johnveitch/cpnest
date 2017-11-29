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

    @classmethod
    def log_likelihood(cls,p):
        return -0.5*(p['x']**2 + p['y']**2) - np.log(2.0*np.pi)


class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(GaussianModel(),verbose=1,Nthreads=4,Nlive=1000,maxmcmc=5000,Poolsize=1000)

    def test_run(self):
        self.work.run()
        # 2 sigma tolerance
        tolerance = 2.0*np.sqrt(self.work.NS.state.info/self.work.NS.Nlive)
        self.assertTrue(np.abs(self.work.NS.logZ - GaussianModel.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0} instead of {1}'.format(self.work.NS.logZ ,GaussianModel.analytic_log_Z))

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
 
