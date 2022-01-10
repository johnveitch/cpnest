# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, boundscheck=False, wraparound=False, binding=True, embedsignature=True

from __future__ import division
from numpy.math cimport INFINITY
cimport numpy as np
import numpy as np

def rebuild_livepoint(list names):
    return _rebuild_livepoint(names)

cdef LivePoint _rebuild_livepoint(list names):
    cdef LivePoint lp = LivePoint(names)
    return lp

cdef class LivePoint:
    """
    Class defining a live point.
    Initialisation arguments:
    names: names of parameters
    d: (optional) an array for initialisation
    logL: (optional) log likelihood of this sample
    logP: (optional) log prior of this sample
    """
    def __cinit__(self,
                  list names,
                  np.ndarray d=None,
                  double logL=-INFINITY,
                  double logP=-INFINITY):
        self.logL = logL
        self.logP = logP
        self.names = names
        self.dimension = len(names)
        if d is not None:
            self.values             = np.array(d, dtype=np.float64)
        else:
            self.values             = np.zeros(self.dimension, dtype=np.float64)

    def keys(self):
        return self.names

    def __reduce__(self):
        return (rebuild_livepoint, (self.names,),self.__getstate__())
    
    def __repr__(self):
        return self.__class__.__name__+'({0:s}, d={1:s}, logL={2:f}, logP={3:f})'.format(str(self.names),str(self.values),self.logL,self.logP)

    def __getstate__(self):
        return (self.logL, self.logP, self.values)
    
    def __setstate__(self, state):
        self.logL   = state[0]
        self.logP   = state[1]
        self.values = np.array(state[2], dtype=np.float64)

    def __str__(LivePoint self):
        return str({n:self[n] for n in self.names})

    def __cmp__(LivePoint self, LivePoint other):
        cdef Py_ssize_t i
        for i in range(self.dimension):
            if not self.names[i] in other.names or self[self.names[i]]!=other[self.names[i]]:
                return 1
        return 0

    def __add__(LivePoint self, LivePoint other):
        assert self.dimension == other.dimension
        cdef LivePoint result = LivePoint(self.names)
        result.values = self.values+other.values
        return result

    def __iadd__(LivePoint self, LivePoint other):
        assert self.dimension == other.dimension
        self.values += other.values
        return self
    
    def __sub__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        cdef LivePoint result = LivePoint(self.names)
        result.values = self.values-other.values
        return result

    def __isub__(LivePoint self, LivePoint other):
        assert self.dimension == other.dimension
        self.values -= other.values
        return self

    def __mul__(LivePoint self, double other):
        cdef LivePoint result = LivePoint(self.names)
        result.values = other*self.values
        return result

    def __imul__(LivePoint self, double other):
        self.values *= other
        return self

    def __truediv__(LivePoint self, double other):
        cdef LivePoint result = LivePoint(self.names)
        result.values = self.values/other
        return result

    def __itruediv__(LivePoint self, double other):
        self.values /= other
        return self

    def __len__(LivePoint self):
        return self.dimension
    
    def __getitem__(LivePoint self, str name):
        cdef Py_ssize_t i
        for i in range(self.dimension):
            if self.names[i] == name:
                return self.values[i]
        raise KeyError

    def __setitem__(LivePoint self, str name, double value):
        cdef Py_ssize_t i
        for i in range(self.dimension):
            if self.names[i] == name:
                self.values[i] = value
                return
        raise KeyError
    
    def copy(LivePoint self):
        return self._copy()
    
    cdef LivePoint _copy(LivePoint self):
        """
        Returns a copy of the current live point
        """
        cdef LivePoint result = LivePoint(self.names)
        result.__setstate__(self.__getstate__())
        return result
    
    def asnparray(LivePoint self):
        return self._asnparray()
    
    cdef np.ndarray _asnparray(LivePoint self):
        """
        Return the sample as a numpy record array
        """
        cdef list names = self.names+['logL','logPrior']
        cdef np.ndarray x = np.zeros(1, dtype = {'names':names, 'formats':['f8' for _ in names]})
        for n in self.names:
            x[n] = self[n]
        x['logL']     = self.logL
        x['logPrior'] = self.logP
        return x
