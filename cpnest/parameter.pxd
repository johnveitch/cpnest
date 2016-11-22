from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport log,exp,sqrt,cos

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cdef class LivePoint:
    cdef public double logL
    cdef public double logP
    cdef public unsigned int dimension
    cdef public list parameters
    cdef public list names
    #def __getitem__(self, str name)
    cpdef double get(self, str name)
    #def void __setitem__(self, str name, double value)
    cpdef void set(self, str name, double value)

