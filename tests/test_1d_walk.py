#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from scipy import integrate,stats
from scipy.special import erf
import cpnest
import cpnest.model

class GaussianModel(cpnest.model.Model):
    """
    A simple gaussian model with parameters mean and sigma
    """
    def __init__(self, mean=0., sigma=1.0, bounds=[[0., 10]]):
        self.distr = stats.norm(loc=mean,scale=sigma)
        self.mean = mean
        self.sigma = sigma
        self.bounds = bounds
        inv_sqrt2 = 1./np.sqrt(2.)
        minv = self.bounds[0][0]
        maxv = self.bounds[0][1]
        lnC = -np.log(maxv - minv)
        self._analytic_log_Z = np.log(0.5*(erf(inv_sqrt2*(self.mean - minv)/self.sigma) - erf(inv_sqrt2*(self.mean - maxv)/self.sigma)))
        self._analytic_log_Z += lnC # divide by prior

        D = np.sqrt(2.*np.pi*self.sigma*self.sigma)
        Cpart = 0.25*(1. + 2.*lnC + 2.*np.log(D));
        self._KLdiv = (0.5/D)*np.exp(-0.5*((maxv-self.mean)/self.sigma)*((maxv-self.mean)/self.sigma))*(maxv-self.mean) - Cpart*erf(inv_sqrt2*(maxv-self.mean)/self.sigma)
        self._KLdiv -= ((0.5/D)*np.exp(-0.5*((minv-self.mean)/self.sigma)*((minv-self.mean)/self.sigma))*(minv-self.mean) - Cpart*erf(inv_sqrt2*(minv-self.mean)/self.sigma))
    
    names=['x']

    def log_likelihood(self,p):
        return self.distr.logpdf(p['x'])
        #return -0.5*(p['x']**2) - 0.5*np.log(2.0*np.pi)
        
    @property
    def analytic_log_Z(self):
        return self._analytic_log_Z
    
    @property
    def KLdiv(self):
        return self._KLdiv

bounds = [[0., 1e8]]
mean = 0.
sigma = 1.0

proposals = ['EnsembleWalk']
proposalweights = [1.0]

Nlive = 512
model = GaussianModel(mean=mean, sigma=sigma, bounds=bounds)

logZs = []
logZratio = []
errs = []

for i in range(20):
  work = cpnest.CPNest(model,verbose=0,Nthreads=3,Nlive=Nlive,maxmcmc=2000,poolsize=Nlive,proposals=proposals, proposalweights=proposalweights)
  work.run()

  errs.append(np.sqrt(work.NS.state.info/work.NS.Nlive))
  logZs.append(work.NS.logZ)
  logZratio.append(work.NS.logZ-model.analytic_log_Z)
  #print('2-sigma statistic error in logZ: {0:0.3f}'.format(tolerance))
  #print('Analytic logZ {0}'.format(model.analytic_log_Z))
  #print('Estimated logZ {0}'.format(work.NS.logZ))
  #print('KL divergence {0:0.3f}'.format(model.KLdiv))

# get posterior samples
#pos = work.posterior_samples['x']

