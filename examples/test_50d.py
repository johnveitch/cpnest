import unittest
import numpy as np
import cpnest.model

class GaussianModel(cpnest.model.Model):
    """
    An n-dimensional gaussian
    """
    def __init__(self,dim=50):
        self.dim=dim
        self.names=['{0}'.format(i) for i in range(self.dim)]
        self.bounds=[[-10,10] for _ in range(self.dim)]
        self.analytic_log_Z=0.0 - sum([np.log(bounds[i][1]-bounds[i][0]) for i in range(self.dim)])

    def log_likelihood(self,p):
        return -0.5*(sum(x**2 for x in p.values)) - 0.5*self.dim*np.log(2.0*np.pi)

class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(GaussianModel(),verbose=1,Nthreads=4,Nlive=1000,maxmcmc=5000)

    def test_run(self):
        self.work.run()
        # 2 sigma tolerance
        tolerance = 2.0*np.sqrt(self.work.NS.state.info/self.work.NS.Nlive)
        self.assertTrue(np.abs(self.work.NS.logZ - GaussianModel.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0} instead of {1}'.format(self.work.NS.logZ ,GaussianModel.analytic_log_Z))

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
 
