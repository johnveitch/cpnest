import unittest
import numpy as np
import cpnest
from scipy.special import logsumexp

class GaussianMixtureModel(object):
    """
    A simple gaussian model with parameters mean and sigma
    """
    def __init__(self):
        pass
    par_names=['mean1','sigma1','mean2','sigma2','weight']
    bounds=[[-10,10],[0.01,1],[-10,10],[0.01,1],[0.0,1.0]]
    data = []
    for _ in range(1000):
        if np.random.uniform(0.0,1.0) < 0.3:
            data.append(np.random.normal(0.5,0.5))
        else:
            data.append(np.random.normal(-1.5,0.01))

    @classmethod
    def log_likelihood(cls,x):
        w = x('weight')
        logL = 0.0
        for d in data:
            logL += np.log(w*np.exp(-0.5*np.sum((cls.data-x('mean1'))**2/x('sigma1')**2))/np.sqrt(len(cls.data)*np.log(x('sigma')) - 0.5*np.log(2.0*np.pi)
        
        
        
        return -0.5*np.sum((cls.data-x('mean1'))**2/x('sigma1')**2) - len(cls.data)*np.log(x('sigma')) - 0.5*np.log(2.0*np.pi)-1000

    @staticmethod
    def log_prior(p):
        for i in range(p.dimension):
            if not p.parameters[i].inbounds(): return -np.inf
        return -np.log(p('sigma1'))-np.log(p('sigma2'))


class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(GaussianModel,verbose=1,Nthreads=8,Nlive=1000,maxmcmc=1024)

    def test_run(self):
        self.work.run()



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
