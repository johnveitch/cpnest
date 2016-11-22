import unittest
import numpy as np
import cpnest

class GaussianModel(object):
    """
    A simple gaussian model with parameters mean and sigma
    """
    def __init__(self):
        pass
    par_names=['mean','sigma']
    bounds=[[-10,10],[0.01,1]]
    data = np.array([x for x in np.random.normal(0.5,0.5,size=1000)])

    @classmethod
    def log_likelihood(cls,x):
        return -0.5*np.sum((cls.data-x['mean'])**2/x['sigma']**2) - len(cls.data)*np.log(x['sigma']) - 0.5*np.log(2.0*np.pi)-1000
    @staticmethod
    def log_prior(p):
        for i in range(p.dimension):
            if not p.parameters[i].inbounds(): return -np.inf
        return 0


class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(GaussianModel,verbose=1,Nthreads=8)

    def test_run(self):
        self.work.run()



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)

