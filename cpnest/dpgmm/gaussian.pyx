# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from __future__ import division
cimport numpy as np
import numpy as np
from libc.math cimport log,exp,pow,sqrt
cimport cython

cdef extern from "math.h" nogil:
    double log(double x)
    double exp(double x)
    double pow(double x, double y)
    double sqrt(double x)

cdef class Gaussian(object):
    """A basic multivariate Gaussian class. Has caching to avoid duplicate calculation."""
    cdef public np.ndarray mean
    cdef public np.ndarray precision
    cdef public np.ndarray covariance
    cdef public np.ndarray cholesky
    cdef public double norm
    cdef public int dimension
    
    cdef int mean_set
    cdef int precision_set
    cdef int covariance_set
    cdef int norm_set
    cdef int cholesky_set
    
    def __cinit__(self, dims):
        """dims is the number of dimensions. Initialises with mu at the origin and the identity matrix for the precision/covariance. dims can also be another Gaussian object, in which case it acts as a copy constructor."""
        if isinstance(dims, Gaussian):
            self.dimension = dims.dimension.copy()
            self.mean = dims.mean.copy()
            self.precision = dims.precision.copy() if dims.precision!=None else None
            self.covariance = dims.covariance.copy() if dims.covariance!=None else None
            self.norm = dims.norm
            self.cholesky = dims.cholesky.copy() if dims.cholesky is not None else None
        else:
            self.dimension = dims
            self.mean = np.zeros(dims, dtype=np.double)
            self.precision = np.identity(dims, dtype=np.double)
            self.covariance = np.identity(dims, dtype=np.double)
            self.norm = 1.0
            self.cholesky = np.identity(dims, dtype=np.double)

    cpdef setMean(self, np.ndarray[double, ndim=1] mean):
        """
        Sets the mean - you can use anything numpy will interprete as a 1D array of the correct length.
        """
        self.mean = mean
        self.mean_set = 1

    cpdef setPrecision(self, np.ndarray[double, ndim=2] precision):
        """Sets the precision matrix. Alternatively you can use the setCovariance method."""
        self.precision = precision
        self.precision_set = 1

    cpdef setCovariance(self, np.ndarray[double, ndim=2] covariance):
        """Sets the covariance matrix. Alternatively you can use the setPrecision method."""
        self.covariance = covariance
        self.covariance_set = 21

    cpdef np.ndarray[double, ndim=1] getMean(self):
        """Returns the mean."""
        return self.mean

    cpdef np.ndarray[double, ndim=2] getPrecision(self):
        """Returns the precision matrix."""
        if self.precision_set == 0:
            self.precision = np.linalg.inv(self.covariance)
        return self.precision

    cpdef np.ndarray[double, ndim=2] getCovariance(self):
        """Returns the covariance matrix."""
        if self.covariance_set == 0:
            self.covariance = np.linalg.inv(self.precision)
        return self.covariance

    cpdef double getNorm(self):
        """Returns the normalising constant of the distribution. Typically for internal use only."""
        if self.norm_set == 0:
            self.norm = pow(2.0*np.pi,-0.5*self.mean.shape[0]) * sqrt(np.linalg.det(self.getPrecision()))
            self.norm_set = 1
        return self.norm

    cpdef double prob(self, np.ndarray[double, ndim=1] x):
        """Given a vector x evaluates the probability density function at that point."""
        cdef np.ndarray[double, ndim=1] offset = x - self.mean
        cdef double val = np.dot(offset,np.dot(self.getPrecision(),offset))
        return self.getNorm() * exp(-0.5 * val)

    cpdef double logprob(self, np.ndarray[double, ndim=1] x):
        """Given a vector x evaluates the log probability density function at that point."""
        cdef np.ndarray[double, ndim=1] offset = x - self.mean
        cdef double val = np.dot(offset,np.dot(self.getPrecision(),offset))
        return log(self.getNorm()) -0.5 * val

    cpdef np.ndarray[double, ndim=1] sample(self):
        """
        Draws and returns a sample from the distribution.
        """
        if self.cholesky_set == 0:
            self.cholesky = np.linalg.cholesky(self.getCovariance())
            self.cholesky_set = 1
        cdef np.ndarray[double, ndim=1] z = np.random.normal(size=self.dimension)
        return self.mean + np.dot(self.cholesky,z)

    def __str__(self):
        return '{mean:%s, covar:%s}'%(str(self.mean), str(self.getCovariance()))
