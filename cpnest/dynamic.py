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


class Interval(object):
    def __init__(self,a,b,n=0,parent=None):
        if b<a: a,b = b,a
        self.a=a
        self.b=b
        self.children=None
        self.parent=parent
        self.n=n
        self.data=(None,None)
    
    def logt(self):
        if self.n==0:
            return 0
        else:
            return -1.0/(self.n)
    
    def readout(self):
        """
        Returns logX,logL
        """
        logX=[0]
        logL=[self.a]
        for i in self:
            logX.append(logX[-1]+i.logt())
            logL.append(i.b)
        return (np.array(logX),np.array(logL))
    
    def logZ(self):
        """
        Return integral
        """
        logX,logL = self.readout()
        return log_integrate_log_trap(logL[:-1], logX[:-1])

    
    def integrate(self,func):
        """
        Integrate the function over the interval
        """
        return np.sum(
            [ 0.5*(func(i.b)+func(i.a))*(i.logX()) for i in self]
            )
        
    def __contains__(self,x):
        if isinstance(x,Interval):
            return self.a <= x.a <= x.b <= self.b
        return self.a <= x <= self.b
    
    def __str__(self):
        return "({0}, {1}): n={2}".format(self.a,self.b,self.n)
    
    def print_tree(self,pref=''):
        print(pref+str(self))
        if self.children:
            for c in self.children:
                c.print_tree(pref=pref+'\t')
    
    def __iter__(self):
        return next(self)
    
    def __next__(self):
        """
        Iterate over intervals
        """
        if self.children is not None:
            for c in self.children:
                for z in c:
                    yield z
        else:
            yield self

    def tree(self):
        """
        iterate over the tree
        """
        if self.children is not None:
            for c in self.children:
                for z in c.tree():
                    yield z
        else: yield self
        return

    #python 2 compatibility
    def next(self): return self.__next__()

    def find(self,x):
        """
        Return sub-interval containing x
        """
        if not x in self: return None
        if self.children is not None:
            for c in self.children:
                if x in c: return c.find(x)
        else:
            return self
    
    def split(self,x):
        if x in self:
            return [Interval(self.a,x,n=self.n,parent=self),Interval(x,self.b,n=self.n,parent=self)]
        else:
            return [self]
    
    def insert_point(self,x):
        """
        Insert a point into this interval tree
        """
        if self.a==x or self.b==x: return
        if not x in self:
            return
        if self.children is not None:
            for c in self.children:
                if x in c:
                    c.insert_point(x)
        else:
            self.children=self.split(x)

    def insert_interval(self,i):
        """
        Add an interval into the tree
        """
        # Subdivide left and right intervals with beginning and end of i
        self.insert_point(i.a)
        self.insert_point(i.b)
        # Find all intervals intersecting i and increase their multiplicity
        cur = self.find(i.a)
        for cur in self:
            if cur in i:
                cur.n+=1
    
    def __add__(self,other):
        if self.b == other.a: return Interval(self.a,other.b)
        elif self.a == other.b : return Interval(other.a,self.b)
        else: raise Exception("Cannot add non-contiguous intervals ({0},{1}), ({2},{3})".format(self.a,self.b,other.a,other.b))

    def points(self):
        yield self.a
        for i in self:
            yield i.b
