import unittest
import numpy as np
from scipy import integrate,stats
import cpnest
import cpnest.model
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

class GaussianModel(cpnest.model.Model):
    """
    A simple gaussian model with parameters mean and sigma
    """
    def __init__(self):
        self.distr = stats.norm(loc=0,scale=1.0)
    names=['x']
    bounds=[[-10,10]]
    analytic_log_Z=0.0 - np.log(bounds[0][1] - bounds[0][0])

    def log_likelihood(self,p):
        return self.distr.logpdf(p['x'])
        #return -0.5*(p['x']**2) - 0.5*np.log(2.0*np.pi)
        
    def force(self, p):
        f = np.zeros(1, dtype = {'names':p.names, 'formats':['f8' for _ in p.names]})
        return f

class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.model=GaussianModel()
        self.work=cpnest.CPNest(self.model,verbose=1,nlive=1000,nthreads=2,maxmcmc=200,poolsize=100)
        self.work.run()

    def test_evidence(self):
        # 2 sigma tolerance
        tolerance = 2.0*np.sqrt(self.work.NS.state.info/self.work.NS.Nlive)
        print('2-sigma statistic error in logZ: {0:0.3f}'.format(tolerance))
        print('Analytic logZ {0}'.format(self.model.analytic_log_Z))
        print('Estimated logZ {0}'.format(self.work.NS.logZ))
        pos=self.work.posterior_samples['x']
        #t,pval=stats.kstest(pos,self.model.distr.cdf)
        stat,pval = stats.normaltest(pos.T)
        print('Normal test p-value {0}'.format(str(pval)))
        plt.figure()
        plt.hist(pos.ravel(),density=True)
        x=np.linspace(self.model.bounds[0][0],self.model.bounds[0][1],100)
        plt.plot(x,self.model.distr.pdf(x))
        plt.title('NormalTest pval = {0}'.format(pval))
        plt.savefig('posterior.png')
        plt.figure()
        plt.plot(pos.ravel(),',')
        plt.title('chain')
        plt.savefig('chain.png')
        self.assertTrue(np.abs(self.work.NS.logZ - GaussianModel.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0:.3f} instead of {1:.3f}'.format(self.work.NS.logZ,GaussianModel.analytic_log_Z ))
        self.assertTrue(pval>0.01,'Normaltest test failed: KS stat = {0}'.format(pval))



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)

