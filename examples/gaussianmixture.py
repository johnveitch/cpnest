import unittest
import numpy as np
import cpnest

class GaussianMixtureModel(object):
    """
    A simple gaussian model with parameters mean and sigma
    """
    def __init__(self):
        pass
    par_names=['mean1','sigma1','mean2','sigma2','weight']
    bounds=[[-3,3],[0.01,1],[-3,3],[0.01,1],[0.0,1.0]]
    data = []
    for _ in range(1000):
        if np.random.uniform(0.0,1.0) < 0.3:
            data.append(np.random.normal(0.5,0.5))
        else:
            data.append(np.random.normal(-1.5,0.03))

    @classmethod
    def log_likelihood(cls,x):
        w = x['weight']
        logL = np.sum([np.logaddexp(np.log(w)-0.5*(d-x['mean1']/x['sigma1'])**2,np.log(1.0-w)-0.5*(d-x['mean2']/x['sigma2'])**2) for d in cls.data])
        return logL-50000.0

    @staticmethod
    def log_prior(p):
        for i in range(p.dimension):
            if not p.parameters[i].inbounds(): return -np.inf
        return -np.log(p['sigma1'])-np.log(p['sigma2'])

def normal(x,m,s):
    d = (x-m)/s
    return np.exp(-0.5*d**2)/np.sqrt(2.0*np.pi*s*s)


class GaussianMixtureTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(GaussianMixtureModel,verbose=1,Nthreads=8,Nlive=1024,maxmcmc=1024)

    def test_run(self):
        self.work.run()



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
