import sys
import unittest
import numpy as np
from scipy import integrate,stats
import cpnest
import cpnest.model
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

class HalfGaussianModel(cpnest.model.Model):
    """
    A simple gaussian model with parameters mean and sigma
    """
    def __init__(self,xmax=10.0):
        self.distr = stats.norm(loc=0,scale=1.0)
        self.bounds=[[0,xmax]]
        self.names=['x']
        self.analytic_log_Z = np.log(self.distr.cdf(self.bounds[0][1]) - self.distr.cdf(self.bounds[0][0])) \
                - np.log(self.bounds[0][1]-self.bounds[0][0])

    def log_likelihood(self,p):
        return self.distr.logpdf(p['x'])

    def force(self,x):
        return np.zeros(1, dtype = {'names':x.names, 'formats':['f8' for _ in x.names]})

class HalfGaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self,xmax=20):
        self.model=HalfGaussianModel(xmax=xmax)
        self.work=cpnest.CPNest(self.model,verbose=1,nlive=500,nthreads=1)
        self.work.run()

    def test_evidence(self):
        # 2 sigma tolerance
        tolerance = 2.0*np.sqrt(self.work.NS.state.info/self.work.NS.Nlive)
        print('2-sigma statistic error in logZ: {0:0.3f}'.format(tolerance))
        print('Analytic logZ {0}'.format(self.model.analytic_log_Z))
        print('Estimated logZ {0}'.format(self.work.NS.logZ))
        pos=self.work.posterior_samples['x']
        #t,pval=stats.kstest(pos,self.model.distr.cdf)
        pos=self.work.posterior_samples['x']
        #t,pval=stats.kstest(pos,self.model.distr.cdf)
        plt.figure()
        plt.hist(pos.ravel(),density=True)
        x=np.linspace(self.model.bounds[0][0],self.model.bounds[0][1],100)
        plt.plot(x,2*self.model.distr.pdf(x))
        plt.savefig('posterior.png')
        plt.figure()
        plt.plot(pos.ravel(),',')
        plt.title('chain')
        plt.savefig('chain.png')
        self.assertTrue(np.abs(self.work.NS.logZ - self.model.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0:.3f} instead of {1:.3f}'.format(self.work.NS.logZ,self.model.analytic_log_Z ))


def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main()

