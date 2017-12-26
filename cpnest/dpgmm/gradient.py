from __future__ import division
import numpy as np
from dpgmm import *
from scipy.misc import logsumexp

class Potential(DPGMM):

    def __init__(self, dimension, samples, *args, **kwargs):
        super(Potential, self).__init__(dimension, *args, **kwargs)
        self.dimension = dimension

        for point in samples:
            self.add(np.atleast_1d(point))

        self.setPrior()
        self.setThreshold(1e-4)
        self.setConcGamma(1,1)
        self.solveGrow(**kwargs)
        self.w, self.components = self.sampleMixture()
    
    def logprob(self, x):
        return logsumexp([c.logprob(x) for c in self.components],b=self.w)
#        return logsumexp([np.log(self.density[0][ind])+prob.logprob(x) for ind,prob in enumerate(self.density[1])])

    def force(self, x):
        """
        computes the force as -\partial V, with V the potential (-log density)
        the derivative is computed analytically assuming from the gaussian mixture model
        that approximates the target distribution p

        if p = \sum_c w_c p_c(x)
        and p_c(x) = exp(-0.5 \sum_{ij} \Lambda_{ij} (x_i-m_i) (x_j-m_j)
        we have V = -log(\sum_c w_c p_c(x))
        then:
        dV/dxi = (1/p) \sum_c w_c p_c(x) \sum_l \Lambda^c_{il} (x^c_l-m^c_l)

        return dV/dx = (dV/dxi for i in dimension)
        """

        x = np.atleast_1d(x)
        p = np.exp(self.logprob(x))
        grad = np.zeros(len(x))
        
        for k in range(len(x)):
            grad[k] += np.sum([self.w[i]*c.prob(x)*np.dot(c.getPrecision()[k,:],x[k]-c.getMean()[k]) for i,c in enumerate(self.components)])

        return grad/p

    def __call__(self, x):
        """
        computes the potential as -log probability
        """
        return -self.logprob(x)

if __name__=="__main__":
    w = 0.3
    s = []
    for _ in range(1000):
        if np.random.uniform() < w:
            s.append(np.random.normal(0,1))
        else:
            s.append(np.random.normal(-5,0.1))
    c = Potential(1, s)

    from pylab import *
    n, dx = np.histogram(s, bins=64, normed=True)
    x = 0.5*(dx[1:]+dx[:-1])
    plot(x,-np.log(n), label = 'samples')
    x = np.linspace(-10,10,1000)
    plot(x,[c(xi) for xi in x], label ='potential')
    plot(x,[c.force(xi) for xi in x], label='force')

    axhline(0.0,color='k')
    legend()
    show()
