from cpython cimport array

cdef class LivePoint:
    cdef public double logL
    cdef public double logP
    cdef public unsigned int dimension
    cdef public array.array values
    cdef public list names
    cdef public list bounds
    cpdef copy(LivePoint self)
    cpdef asnparray(LivePoint self)
