import unittest
import numpy as np
import cpnest.model
from scipy import stats

class GaussianModel(cpnest.model.Model):
    """
    An n-dimensional gaussian
    """
    def __init__(self,dim=50):
        self.distr = stats.norm(loc=0,scale=1.0)
        self.dim=dim
        self.names=['{0}'.format(i) for i in range(self.dim)]
        self.bounds=[[-10,10] for _ in range(self.dim)]
#        self.bounds[0] = [-50, 50]
        self.analytic_log_Z=0.0 - sum([np.log(self.bounds[i][1]-self.bounds[i][0]) for i in range(self.dim)])

    def log_likelihood(self,p):
        return np.sum([-0.5*p[n]**2-0.5*np.log(2.0*np.pi) for n in p.names])##np.sum([self.distr.logpdf(p[n]
    
    def log_prior(self,p):
        logP = super(GaussianModel,self).log_prior(p)
#        for n in p.names: logP += -0.5*(p[n])**2
        return logP
    
    def force(self, p):
        f = np.zeros(1, dtype = {'names':p.names, 'formats':['f8' for _ in p.names]})
#        for n in p.names: f[n] = p[n]
        return f

class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.model=GaussianModel(dim = 50)
        self.work=cpnest.CPNest(self.model, verbose=2, nthreads=6, nlive=1000, maxmcmc=100, poolsize=1000, nslice=6)

    def test_run(self):
        self.work.run()
        # 2 sigma tolerance
        tolerance = 2.0*np.sqrt(self.work.NS.state.info/self.work.NS.Nlive)
        self.assertTrue(np.abs(self.work.NS.logZ - self.model.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0} instead of {1}'.format(self.work.NS.logZ ,self.model.analytic_log_Z))

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
 
