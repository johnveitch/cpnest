import unittest
import numpy as np
import cpnest.model

class GaussianMixtureModel(cpnest.model.Model):
    """
    A simple gaussian model with parameters mean and sigma
    Shows example of using your own data
    """
    names=['mean1','sigma1','mean2','sigma2','weight']
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
        logL = np.sum([np.logaddexp(np.log(w/(np.sqrt(2*np.pi)*x['sigma1']))-0.5*((d-x['mean1'])/x['sigma1'])**2,
                                    np.log((1.0-w)/(np.sqrt(2*np.pi)*x['sigma2']))-0.5*((d-x['mean2'])/x['sigma2'])**2) 
                       for d in cls.data ] )
        return logL

    def log_prior(self,p):
        if not self.in_bounds(p): return -np.inf
        return -np.log(p['sigma1'])-np.log(p['sigma2'])

    def force(self,x):
        f = np.zeros(1, dtype = {'names':x.names, 'formats':['f8' for _ in x.names]})
        f['sigma1'] = 1.0
        f['sigma2'] = 1.0
        return f

class GaussianMixtureTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(GaussianMixtureModel(),verbose=1,nthreads=8,nlive=1024,maxmcmc=1024)

    def test_run(self):
        self.work.run()



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
