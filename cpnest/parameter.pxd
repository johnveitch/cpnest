# cython: language_level=3, boundscheck=False, wraparound=False, infer_types=True, embed_signature=True
cimport numpy as np

cdef class LivePoint:
    cdef public double logL
    cdef public double logP
    cdef public unsigned int dimension
    cdef public np.ndarray values
    cdef public list names
    cpdef LivePoint copy(LivePoint self)
    cpdef np.ndarray asnparray(LivePoint self)
