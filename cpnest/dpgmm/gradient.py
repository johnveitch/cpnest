from __future__ import division
import numpy as np
from dpgmm import *
#import copy
from scipy.misc import logsumexp
#import copy_reg
#import types
#import cumulative
#from utils import *


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
        self.density = self.sampleMixture()
    
    def logprob(self, x):
        return logsumexp([np.log(self.density[0][ind])+prob.logprob(x) for ind,prob in enumerate(self.density[1])])
    
    def force(self, x):
        x = np.atleast_1d(x)
        p = np.exp(self.logprob(x))
        grad = np.zeros(len(x))
        
        for k in range(len(x)):
            grad[k] += np.sum([self.density[0][ind]*prob.prob(x)*np.dot(prob.getPrecision()[k,:],x[k]-prob.getMean()[k]) for ind,prob in enumerate(self.density[1])])

        return grad/p

    def __call__(self, x):
        return -self.logprob(x)

if __name__=="__main__":
    w = 0.5
    s = []
    for _ in range(1000):
        if np.random.uniform() < w:
            s.append(np.random.normal(0,1))
        else:
            s.append(np.random.normal(-5,0.01))
    c = Potential(1, s)

    from pylab import *
    x = np.linspace(-10,10,1000)
    plot(x,[c(xi) for xi in x], label='potential')
    plot(x,[c.force(xi) for xi in x], label='force')
    axhline(0.0,color='k')
    legend()
    show()
