from __future__ import division, print_function
import sys
import os
import numpy as np
from numpy import logaddexp, exp, array, log, log1p
from numpy import inf
from . import nest2pos
from .nest2pos import logsubexp, log_integrate_log_trap
from functools import reduce
import itertools as it
from bisect import bisect_left
import time
from scipy.misc import logsumexp

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result

    return timed

class DynamicNestedSampler(object):
    """
    Dynamic nested sampling algorithm
    From Higson, Handley, Hobson, Lazenby 2017
    https://arxiv.org/pdf/1704.03459.pdf
    """
    def __init__(self,G=0.25,Ninit=1000):
        """
        G:      Goal parameter between 0 and 1.
                0: Optimise for evidence calculation
                1: Optimise for posterior sampling
        Ninit:  Initial number of live points
        """
        self.G = G
        self.Ninit = Ninit
        self.nest=Contour()

    def terminate(self):
        """
        Returns True if termination condition met
        """
        pass


    def run(self):
        """
        Run the algorithm
        """
        importance_frac = 0.9 # Minimum fraction of greatest importance to use in refinement
        termination = False
        while not self.terminate():
            # Recalculate importance function
            importance = self.importance()
            maxI_idx = np.argmax(importance)
            maxI = importance[maxI_idx]
            Lmax = self.logLs[maxI_idx]
            Lmin = min( self.logLs[importance>importance_frac*maxI] )
            self.sample_between(Lmin, Lmax)

class LivePoint(object):
    """
    Representation of a basic atom of NS
    """
    def __init__(self,logLmin=-inf,logL=None,params=None,parent=None):
        """
        logLmin : likelihood contour this sample was generated within
        logL    : likelihood of this sample
        params  : parameters
        """
        self.logLmin=logLmin
        self.logL=logL
        self.params=params
        self.parent=parent
    def is_sibling(self,other):
        return self.logLmin < other.logL and self.logL < other.logL

class Contour(object):
    def __init__(self,logLmin=-inf,logP=0,contents=None):
        self.logLmin=logLmin
        self.logP=logP
        self.points=list([]) if contents is None else contents
    
    @property
    def _logls(self):
        return [p.logL for p in self.points]
    
    def contains(self,p):
        return self.logLmin <= p.logL
    def count(self,ps):
        return sum(map(self.contains,ps))
    def add(self,p):
        if p.logL>=self.logLmin:
            i=bisect_left(self._logls,p.logL)
            self.points.insert(i,p)

    def update(self,ps):
        for p in ps: self.add(p)
    
    @property
    def n(self):
        return len(self.points)

    def logt(self, c):
        """
        log volume of self occupied by c
        """
        if not isinstance(c,Contour): c = Contour(logLmin=c)
        num_below = bisect_left(self._logls, c.logLmin)
        #num_below = sum(p.logL>=c.logLmin for p in self.contents)
        return log(self.n-num_below)-log(self.n)
    
    @property
    def nested(self):
        """
        sub-contours for all points
        """
        i=1
        ps=self.points
        while i<len(ps):
            p=ps[i]
            i+=1
            yield Contour(  logLmin = p.logL,
                    logP = self.logP+self.logt(p.logL),
                    contents=ps[i:]
                    )
    def logw(self):
        if self.n<=1: return 0
        else: return self.logt(self._logls[0])+self.logLmin
    @property
    #@timeit
    def logZ(self):
        """
        Evidence inside this contour
        Z(t) = sum_{dead} L_i w_i(t)
        """
        n=self.n
        if n==0:
            return self.logP+self.logLmin
        logX=np.zeros(n+1)
        logX[0]=self.logP
        for i in range(1,n):
            logX[i]=logX[i-1] - 1/(n-i)
        logX[-1]=-np.inf
        ls=np.concatenate(([self.logLmin],self._logls,self._logls[-1:]))
        log_func_sum = logaddexp(ls[:-1], ls[1:]) - np.log(2)
        return log_integrate_log_trap(log_func_sum, logX)
    
    def I_z(self):
        """
        Importance of current set to evidence
        I_z[i] = E[Z_{>i}] / n_i
        """
        return(self.logZ/(1+self.n))
    def I_post(self):
        """
        Importance of current set to posterior
        I_post[i] = L_i w_i
        """
        return self.logw()
    def I(self,G=0.25):
        """
        Importance
        """
        Iz = array([np.exp(c.I_z()) for c in self.nested])
        Ipost = array([np.exp(c.I_post()) for c in self.nested])
        return (1.0-G)*Iz/sum(Iz) + G*Ipost/sum(Ipost)
        
        
