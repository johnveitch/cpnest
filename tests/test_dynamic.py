import unittest
import numpy as np
from scipy import integrate,stats
import cpnest
import cpnest.model
import cpnest.dynamic
from cpnest.dynamic import Interval
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
plt.ion()
import sys
import os
import time

BOUND = 10 # ten sigma
analytic_log_Z=0.0 - np.log(2*BOUND)

def logLtest(x):
    return np.log(x)

def logLtestInv(l):
    return np.exp(l)

import scipy

norm = scipy.stats.norm()

def sample_above(l):
    x = np.sqrt(-np.log(2*np.pi)-2*l)
    #print(l,x)
    return np.random.uniform(-x,x)
    #dx = 2*norm.isf(np.exp(l))
    

logLtest=norm.logpdf

xs = np.random.rand(1000)*BOUND*2 - BOUND

def logLX(logX):
    return logLtest(np.exp(logX))


print('logZ true: {0}'.format(analytic_log_Z))

from cpnest.dynamic import DynamicNestedSampler

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


d = DynamicNestedSampler(GaussianModel())


"""


i = Interval(logLtest(-50),np.inf)
for x in xs[1:]:
    logL=logLtest(x)
    i.insert_interval(Interval(-np.inf,logL))
    #i.insert_interval(Interval(logL,np.inf))

import itertools as it
it=1
logZs=[]
while it<10000:
    logliter = [x.b for x in i]
    N=len(logliter)
    for logLmin in logliter[:N//3]:
        #logLmin = p.b
        #print(logLmin)
        if np.isinf(logLmin) or np.isnan(logLmin): break
        newx = sample_above(logLmin)
        i.insert_interval(Interval(logLmin,logLtest(newx)))
        logZs.append(i.logZ())
        print('{2}: {0:.3f} -> {1:.3f} logZ {3:.4f}'.format(logLmin,logLtest(newx),it,logZs[-1]))
        it+=1
        if it%100: continue
        plt.clf()
        plt.plot(logZs-analytic_log_Z)
        plt.grid()
        plt.xlabel('iteration')
        plt.ylabel('logZ error')
        plt.draw()
        plt.pause(0.03)
    
#plt.plot([a.a for a in i],[a.logX() for a in i])
ps=[p for p in i.points()]
M=[x.logt() for x in i]
plt.plot(np.log(M),ps[1:])
"""
