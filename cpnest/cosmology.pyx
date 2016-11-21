from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh
cimport cython

cdef extern from "lal/LALCosmologyCalculator.h" nogil:
    ctypedef struct LALCosmologicalParameters:
        double h;
        double om;
        double ol;
        double ok;
        double w0;
        double w1;
        double w2;
    
    cdef double XLALLuminosityDistance(
            LALCosmologicalParameters *omega, 
            double z)

    cdef double XLALAngularDistance(
            LALCosmologicalParameters *omega, 
            double z)

    cdef double XLALComovingLOSDistance(
            LALCosmologicalParameters *omega, 
            double z)

    cdef double XLALComovingTransverseDistance(
            LALCosmologicalParameters *omega, 
            double z)

    cdef double XLALHubbleDistance(
            LALCosmologicalParameters *omega
            )

    cdef double XLALHubbleParameter(double z,
            void *omega
            )

    cdef double XLALIntegrateHubbleParameter(LALCosmologicalParameters *omega, double z)

    cdef double XLALUniformComovingVolumeDistribution(
            LALCosmologicalParameters *omega, 
            double z,
            double zmax)

    cdef double XLALUniformComovingVolumeDensity(
            double z,
            void *omega)

    cdef double XLALIntegrateComovingVolumeDensity(LALCosmologicalParameters *omega, double z)

    cdef double XLALIntegrateComovingVolume(LALCosmologicalParameters *omega, double z)

    cdef double XLALComovingVolumeElement(double z, void *omega)

    cdef double XLALComovingVolume(LALCosmologicalParameters *omega, double z)

    cdef LALCosmologicalParameters *XLALCreateCosmologicalParameters(double h, double om, double ol, double w0, double w1, double w2)


    cdef void XLALDestroyCosmologicalParameters(LALCosmologicalParameters *omega)

    cdef double XLALGetHubbleConstant(LALCosmologicalParameters *omega)

    cdef double XLALGetOmegaMatter(LALCosmologicalParameters *omega)

    cdef double XLALGetOmegaLambda(LALCosmologicalParameters *omega)

    cdef double XLALGetOmegaK(LALCosmologicalParameters *omega)

    cdef double XLALGetW0(LALCosmologicalParameters *omega)

    cdef double XLALGetW1(LALCosmologicalParameters *omega)

    cdef double XLALGetW2(LALCosmologicalParameters *omega)


cdef class CosmologicalParameters:
    cdef LALCosmologicalParameters* __LALCosmologicalParameters
    cdef public double h
    cdef public double om
    cdef public double ol
    def __cinit__(self,double h, double om, double ol):
        self.h = h
        self.om = om
        self.ol = ol
        self.__LALCosmologicalParameters = XLALCreateCosmologicalParameters(self.h,self.om,self.ol,-1.0,0.0,0.0)
    
    cpdef void SetH(self, double h):
        self.h = h
        self.__LALCosmologicalParameters.h = h
    
    cpdef void SetOM(self, double om):
        self.om = om
        self.__LALCosmologicalParameters.om = om

    cpdef void SetOL(self, double ol):
        self.ol = ol
        self.__LALCosmologicalParameters.ol = ol

    cpdef double HubbleParameter(self,double z):
        return XLALHubbleParameter(z, self.__LALCosmologicalParameters)

    cpdef double LuminosityDistance(self, double z):
        return XLALLuminosityDistance(self.__LALCosmologicalParameters,z)

    cpdef double HubbleDistance(self):
        return XLALHubbleDistance(self.__LALCosmologicalParameters)

    cpdef double IntegrateComovingVolumeDensity(self, double zmax):
        return XLALIntegrateComovingVolumeDensity(self.__LALCosmologicalParameters,zmax)

    cpdef double IntegrateComovingVolume(self, double zmax):
        return XLALIntegrateComovingVolume(self.__LALCosmologicalParameters,zmax)

    cpdef double UniformComovingVolumeDensity(self, double z):
        return XLALUniformComovingVolumeDensity(z, self.__LALCosmologicalParameters)

    cpdef double UniformComovingVolumeDistribution(self, double z, double zmax):
        return XLALUniformComovingVolumeDistribution(self.__LALCosmologicalParameters, z, zmax)

    cpdef double ComovingVolumeElement(self,double z):
        return XLALComovingVolumeElement(z, self.__LALCosmologicalParameters)

    cpdef double ComovingVolume(self,double z):
        return XLALComovingVolume(self.__LALCosmologicalParameters, z)
