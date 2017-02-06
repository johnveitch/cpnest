import unittest
import numpy as np
import cpnest

Nlive = 1000
maxmcmc = 1000

class GaussianModel(object):
    """
    A simple gaussian model with parameters mean and sigma
    """
    def __init__(self):
        pass
    par_names=['x']
    bounds=[[-100,100]]
    analytic_log_Z=0.0

    @classmethod
    def log_likelihood(cls,p):
        return -0.5*p['x']**2 - np.log(2.0*np.pi)

    @staticmethod
    def log_prior(p):
        for i in range(p.dimension):
            if not p.parameters[i].inbounds(): return -np.inf
        return 0.0


class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(GaussianModel,verbose=1,Nthreads=8,Nlive=Nlive,maxmcmc=maxmcmc)

    def test_run(self):
        self.work.run()
        tolerance = 0.5
        NSlogZ = np.loadtxt('chain_%d_1234.txt_evidence.txt'%Nlive)[0]
        self.assertTrue(np.abs(NSlogZ - GaussianModel.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0}'.format(NSlogZ ))



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
 
