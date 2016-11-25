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
    cdef public list bounds

