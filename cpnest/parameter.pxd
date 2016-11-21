from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport log,exp,sqrt,cos

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cdef class parameter:
    cdef public str name
    cdef public double value
    cdef public double bounds[2]
    cpdef inbounds(self)

cdef class LivePoint:
    cdef public double logL
    cdef public double logP
    cdef public unsigned int dimension
    cdef public list parameters
    cdef public list names
    cpdef double get(self, str name)
    cpdef void set(self, str name, double value)

