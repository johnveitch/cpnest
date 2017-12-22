from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport log,exp

DTYPE = np.float

ctypedef np.float_t DTYPE_t

cdef inline double d_max(double a, double b): return a if a >= b else b

cdef inline double log_add(double x, double y): return x+log(1+exp(y-x)) if x >= y else y+log(1+exp(x-y))

cpdef fast_log_cumulative(np.ndarray[DTYPE_t, ndim=1]  f):
    cdef size_t n = f.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] h = np.zeros(n, dtype=DTYPE)

    h[0] = f[0]
    for i in range(1,n):
        h[i] = log_add(h[i-1],f[i])
    return h-h[-1]
