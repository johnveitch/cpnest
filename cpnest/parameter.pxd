# cython: language_level=3, boundscheck=False, wraparound=False, infer_types=True, embed_signature=True
cimport numpy as np

cdef class LivePoint:
    cdef public double logL
    cdef public double logP
    cdef public unsigned int dimension
    cdef public np.ndarray values
    cdef public list names
    cdef LivePoint _copy(LivePoint self)
    cdef np.ndarray _asnparray(LivePoint self)
