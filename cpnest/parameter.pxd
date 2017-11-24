from cpython cimport array
cimport numpy as np

cdef class LivePoint:
    cdef public double logL
    cdef public double logP
    cdef public unsigned int dimension
    cdef public array.array values
    cdef public list names
    cdef public list bounds
    cpdef LivePoint copy(LivePoint self)
    cpdef np.ndarray asnparray(LivePoint self)
