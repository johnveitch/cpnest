# encoding: utf-8
# cython: profile=False
# cython: linetrace=False

from __future__ import division
from numpy import inf
from cpython cimport array
import numpy as np

def rebuild_livepoint(names):
  lp=LivePoint(names)
  return lp

cdef class LivePoint:
    def __cinit__(LivePoint self, list names, d=None, logL=-inf, logP=-inf):
        self.logL = logL
        self.logP = logP
        self.names = names
        self.dimension = len(names)
        if d is not None:
          self.values = array.array('d',d)
        else:
          self.values = array.array('d',[0]*self.dimension)

    def __reduce__(self):
        return (rebuild_livepoint, (self.names,),self.__getstate__()) 
    
    def __repr__(self):
        return 'LivePoint({0:s}, d={1:s}, logL={2:f}, logP={3:f})'.format(str(self.names),str(self.values),self.logL,self.logP)

    def __getstate__(self):
      return (self.logL,self.logP,self.values)
    def __setstate__(self,state):
      self.logL=state[0]
      self.logP=state[1]
      self.values=array.array('d',state[2])

    def __str__(LivePoint self):
        return str({n:self[n] for n in self.names})

    def __cmp__(LivePoint self,LivePoint other):
        for i in range(self.dimension):
            if not self.names[i] in other.names or self[self.names[i]]!=other[self.names[i]]:
                return 1
        return 0

    def __add__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        result=LivePoint(self.names)
        cdef int i
        for i in range(self.dimension):
          result.values[i]=self.values[i]+other.values[i]
        return result

    def __iadd__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        cdef int i
        for i in range(self.dimension):
            self.values[i]+=other.values[i]
        return self
    
    def __sub__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        result = LivePoint(self.names)
        cdef i
        for i in range(self.dimension):
            result.values[i]=self.values[i]-other.values[i]
        return result

    def __isub__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        cdef int i
        for i in range(self.dimension):
            self.values[i]-=other.values[i]
        return self

    def __mul__(LivePoint self,float other):
        result=LivePoint(self.names)
        cdef int i
        for i in range(self.dimension):
            result.values[i]=other*self.values[i]
        return result

    def __imul__(LivePoint self,float other):
        cdef int i
        for i in range(self.dimension):
            self.values[i]*=other
        return self

    def __truediv__(LivePoint self,float other):
        result = LivePoint(self.names)
        cdef int i
        for i in range(self.dimension):
            result.values[i]=self.values[i]/other
        return result

    def __itruediv__(LivePoint self,float other):
        cdef int i
        for i in range(self.dimension):
            self.values[i]/=other
        return self

    def __len__(LivePoint self):
        return self.dimension
    
    def __getitem__(LivePoint self, str name):
        cdef unsigned int i
        for i in range(self.dimension):
            if self.names[i] == name:
                return self.values[i]
        raise KeyError

    def __setitem__(LivePoint self, str name, double value):
        cdef unsigned int i
        for i in range(self.dimension):
            if self.names[i] == name:
                self.values[i] = value
                return
        raise KeyError

    cpdef copy(LivePoint self):
      result = LivePoint(self.names)
      result.__setstate__(self.__getstate__())
      return result
      
    cpdef asnparray(LivePoint self):
      names = self.names+['logL','logPrior']
      x = np.zeros(1, dtype = {'names':names, 'formats':['f8' for _ in names]})
      for n in self.names: x[n] = self[n]
      x['logL'] = self.logL
      x['logPrior'] = self.logP
      return x
                

