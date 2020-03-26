# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, boundscheck=False, wraparound=False

from __future__ import division
from numpy.math cimport INFINITY
from cpython cimport array
cimport numpy as np
import numpy as np

cpdef LivePoint rebuild_livepoint(names, bounds):
    cdef LivePoint lp=LivePoint(names, bounds)
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
    def __cinit__(LivePoint self,
                  list names,
                  list bounds,
                  array.array d=None,
                  double logL=-INFINITY,
                  double logP=-INFINITY):
        self.logL = logL
        self.logP = logP
        self.names = names
        self.bounds = bounds
        self.dimension = len(names)
        if d is not None:
            self.values             = array.array('d',d)
        else:
            self.values             = array.array('d',[0]*self.dimension)

    def keys(self):
        return self.names

    def __reduce__(self):
        return (rebuild_livepoint, (self.names,self.bounds),self.__getstate__())
    
    def __repr__(self):
        return self.__class__.__name__+'({0:s}, d={1:s}, logL={2:f}, logP={3:f})'.format(str(self.names),str(self.values),self.logL,self.logP)

    def __getstate__(self):
        return (self.logL,self.logP,self.bounds,self.values)
    
    def __setstate__(self,state):
          self.logL=state[0]
          self.logP=state[1]
          self.bounds=state[2]
          self.values=array.array('d',state[3])

    def __str__(LivePoint self):
        return str({n:self[n] for n in self.names})

    def __cmp__(LivePoint self, LivePoint other):
        cdef Py_ssize_t i
        for i in range(self.dimension):
            if not self.names[i] in other.names or self[self.names[i]]!=other[self.names[i]]:
                return 1
        return 0

    def __add__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        result=LivePoint(self.names,self.bounds)
        cdef Py_ssize_t i
        for i in range(self.dimension):
            result.values[i]=self.values[i]+other.values[i]
        return result

    def __iadd__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        cdef Py_ssize_t i
        for i in range(self.dimension):
            self.values[i]+=other.values[i]
        return self
    
    def __sub__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        cdef LivePoint result = LivePoint(self.names,self.bounds)
        cdef Py_ssize_t i
        for i in range(self.dimension):
            result.values[i]=self.values[i]-other.values[i]
        return result

    def __isub__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        cdef Py_ssize_t i
        for i in range(self.dimension):
            self.values[i]-=other.values[i]
        return self

    def __mul__(LivePoint self,float other):
        cdef LivePoint result=LivePoint(self.names,self.bounds)
        cdef Py_ssize_t i
        for i in range(self.dimension):
            result.values[i]=other*self.values[i]
        return result

    def __imul__(LivePoint self,float other):
        cdef Py_ssize_t i
        for i in range(self.dimension):
            self.values[i]*=other
        return self

    def __truediv__(LivePoint self,float other):
        cdef LivePoint result = LivePoint(self.names,self.bounds)
        cdef Py_ssize_t i
        for i in range(self.dimension):
            result.values[i]=self.values[i]/other
        return result

    def __itruediv__(LivePoint self,float other):
        cdef Py_ssize_t i
        for i in range(self.dimension):
            self.values[i]/=other
        return self

    def __len__(LivePoint self):
        return self.dimension
    
    def __getitem__(LivePoint self, name):
        cdef Py_ssize_t i
        for i in range(self.dimension):
            if self.names[i] == name:
                return self.values[i]
        raise KeyError

    def __setitem__(LivePoint self, name, double value):
        cdef Py_ssize_t i
        for i in range(self.dimension):
            if self.names[i] == name:
                self.values[i] = value
                return
        raise KeyError
    
    cpdef LivePoint copy(LivePoint self):
        """
        Returns a copy of the current live point
        """
        cdef LivePoint result = LivePoint(self.names,self.bounds)
        result.__setstate__(self.__getstate__())
        return result
      
    cpdef np.ndarray asnparray(LivePoint self):
        """
        Return the sample as a numpy record array
        """
        cdef list names = self.names+['logL','logPrior']
        cdef np.ndarray x = np.zeros(1, dtype = {'names':names, 'formats':['f8' for _ in names]})
        for n in self.names: x[n] = self[n]
        x['logL']     = self.logL
        x['logPrior'] = self.logP
        return x

cdef inline double map_to_values(double normalised_value, double lower_bound, double upper_bound):
    """
    Maps [-1,1] to the bounds of the parameters
    """
    return 0.5*(normalised_value+1.0)*(upper_bound-lower_bound)+lower_bound

cdef inline double map_to_normalised_values(double value, double lower_bound, double upper_bound):
    """
    Maps the bounds of the parameters onto [-1,1]
    """
    return (value-lower_bound)/(upper_bound-lower_bound)

