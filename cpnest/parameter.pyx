from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport log,exp,sqrt,cos,fabs,sin
cimport cython
from cosmology import *
from scipy.optimize import newton
import scipy.integrate as integ

""" imports from gsl for fast gamma function evaluation as well as for disabling the gsl error handling that would crash the program """

cdef extern from "gsl/gsl_errno.h":
    ctypedef void gsl_error_handler_t
    int GSL_SUCCESS
    int GSL_EUNDRFLW
    char *gsl_strerror(int gsl_errno)
    gsl_error_handler_t* gsl_set_error_handler_off()

gsl_set_error_handler_off()

cdef extern from "gsl/gsl_sf_gamma.h":
    cdef double gsl_sf_gamma_inc (double a, double x)

cdef extern from "gsl/gsl_randist.h":
    cdef double gsl_cdf_poisson_P (unsigned int k, double mu)

cdef extern from "gsl/gsl_sf_erf.h":
    cdef double gsl_sf_erf(double x)
    cdef double gsl_sf_erfc(double x)
    cdef double gsl_sf_erf_Q(double x)

cdef inline double log_add(double x, double y): return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

cdef double snr_threshold = 8.0

cdef class parameter:

    def __cinit__(self, str name, list bound):
        self.name = name
        self.bounds[0] = bound[0]
        self.bounds[1] = bound[1]
        self.value = np.random.uniform(self.bounds[0],self.bounds[1])

    def __str__(self):
        return 'parameter %s : %s in %s - %s' % (self.name,repr(self.value),repr(self.bounds[0]),repr(self.bounds[1]))

    cpdef inbounds(self):
        if self.value > self.bounds[1] or self.value < self.bounds[0]:
            return False
        return True

cdef class LivePoint:

    def __cinit__(self, list names, list bounds):
        self.logL = -np.inf
        self.logP = -np.inf
        self.names = names
        self.dimension = len(names)
        self.parameters = []
        cdef unsigned int i
        for i in range(self.dimension):
            self.parameters.append(parameter(names[i],bounds[i]))

    cpdef double get(self, str name):
        cdef unsigned int i
        for i in range(self.dimension):
            if self.parameters[i].name == name:
                return self.parameters[i].value

    cpdef void set(self, str name, double value):
        cdef unsigned int i
        for i in range(self.dimension):
            if self.parameters[i].name == name:
                self.parameters[i].value = value

cdef class Event:
    def __cinit__(self, unsigned  int ID, double dl, double sigma, double snr, double domega, double m1, double m2, np.ndarray[double, ndim = 1] redshifts):
        self.n_hosts = redshifts.shape[0]
        self.ID = ID
        self.snr = snr
        self.dl = dl
        self.domega = domega
        self.sigma = sigma
        self.redshifts = redshifts
        self.m1 = m1
        self.m2 = m2
        self.dmax = (self.snr/snr_threshold)*(self.dl+3.0*self.sigma)
        self.dmin = self.dl-3.0*self.sigma
        if self.dmin < 0.0: self.dmin = 0.0


cpdef double find_redshift(object omega, double dl):
    return newton(objective,0.1,args=(omega,dl))

cdef double objective(double z, object omega, double dl):
    return dl - omega.LuminosityDistance(z)

cpdef void copy_live_point(LivePoint out_live, LivePoint in_live):
    """
    helper function to copy live points
    """
    cdef unsigned int i
    for i in range(in_live.dimension):
        out_live.parameters[i].value = np.copy(in_live.parameters[i].value)
    out_live.logL = np.copy(in_live.logL)
    out_live.logP = np.copy(in_live.logP)

cpdef double logPrior(list data, LivePoint x):
    cdef unsigned int i
    cdef double logP = 0.0
    for i in range(x.dimension):
        if not(x.parameters[i].inbounds()):
            logP = -np.inf
            return logP
    cdef double om = x.get('om')
    cdef object omega = CosmologicalParameters(x.get('h'),om,1.0-om)
#cdef double zmax
#    cdef double zmin
#    cdef double Vmax, Vmin, V

    if data is None:
        return log(omega.ComovingVolumeElement(x.get('z0')))

#    cdef unsigned int N = len(data)
#    for i in range(N):
#        zmax = np.max(data[i].redshifts)
#        zmin = np.min(data[i].redshifts)
#        if zmax > zmin:
#            if zmin > 0.0:
#                Vmin = omega.IntegrateComovingVolume(zmin)
#            else:
#                Vmin = 0.0
#            V = omega.IntegrateComovingVolume(zmax)-Vmin
#        logP += log(omega.ComovingVolumeElement(x.get('z%d'%data[i].ID)))#-log(V)
    return logP

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef double logLikelihood(list data, LivePoint x):
    """
    Likelihood function
    """
    if data is None:
        return 0.0
    cdef unsigned int i,j
    cdef unsigned int N = len(data)
    cdef double logL=0.0
    cdef double T = 5.0
    cdef double om = x.get('om')
    cdef object omega = CosmologicalParameters(x.get('h'),om,1.0-om)
    cdef double event_redshift

    for i in range(N):
        event_redshift = x.get('z%d'%data[i].ID)
        logL += logLikelihood_single(data[i].redshifts,data[i].dl,data[i].sigma, omega, data[i].dmin, data[i].dmax, data[i].snr, event_redshift)

    return logL

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double logLikelihood_single(np.ndarray[DTYPE_t, ndim=1] redshifts, double meandl, double sigma, object omega, double dmin, double dmax, double snr, double event_redshift):
    """
    Likelihood function
    """
    cdef unsigned int i
    cdef unsigned int N = redshifts.shape[0]
    cdef double logL_galaxy
    cdef double dl,z
    cdef double score,score_z
    cdef double logL = -np.inf
    cdef double galaxy_snr
    cdef int count = 0
    cdef double proper_motion = 0.0015
    cdef double C = 2.99792458e8

    dl = omega.LuminosityDistance(event_redshift)
    score = (dl-meandl)/sigma
    for i in range(N):
        z = redshifts[i]
            #if meandl-sigma < omega.LuminosityDistance(z) < meandl+sigma:
        score_z = (event_redshift-z)/proper_motion
        logL_galaxy = -0.5*score_z*score_z
        #        else:
        #            logL_galaxy = - np.inf
                #if not(np.isnan(logL_galaxy)) and not(np.isinf(logL_galaxy)):
        logL = log_add(logL,logL_galaxy)
    logL += -0.5*score*score
    return logL-log(N)

#cpdef double selection_function(double dl):
#    cdef double e[4]
#    e[:] = [-3.42261679e-09,7.69511008e-06,-5.80817624e-03,1.49960918e+00]
#    cdef double s = e[0]*dl*dl*dl+e[1]*dl*dl+e[2]*dl+e[3]
#    if s > 1.0: s = 1.0
#    if s < 0.0: s = 0.0
#    return s

cpdef double selection_function(double dl):
#    cdef double e[4]
#    e[:] = [-3.42261679e-09,7.69511008e-06,-5.80817624e-03,1.49960918e+00]
#    cdef double s = e[0]*dl*dl*dl+e[1]*dl*dl+e[2]*dl+e[3]
#    if s > 1.0: s = 1.0
#    if s < 0.0: s = 0.0
    s = 0
    if dl < 500.0: s = 1
    return s


cpdef double integrand_volume_weight(double z, object omega):
    cdef double dl = omega.LuminosityDistance(z)
    return omega.UniformComovingVolumeDensity(z)*selection_function(dl)

cpdef double rate_integral(double zmax, object omega):
    return 1e-9*integ.quad(integrand_volume_weight,0.0,zmax,args = (omega))[0]# in Gpc

cpdef double logLikelihood_nondetection(double dlmax, double Vmax, double Volume_max,int N):
    """
    Likelihood function for the missed detections because not loud enough
    """
    cdef unsigned int i
    cdef double T = 5.0 # hardcoded observation time
    cdef double logL= 0.0
    cdef double rate = 1000.0
    cdef double RVT = rate*T*Vmax/1e9
    cdef int missed_events = int(RVT) - N
    cdef double rho_missed
    for i in range(missed_events):
        while 1:
            dl_missed = np.random.uniform(0.0,Volume_max)**(1.0/3.0)
            rho_missed = dlmax/dl_missed
            if rho_missed < snr_threshold: break
        logL += log(0.5*gsl_sf_erf(rho_missed/sqrt(2)))
    return logL-RVT-missed_events*log(RVT)-missed_events*log(missed_events)+missed_events

cdef double normal(x,m,s):
    cdef double z = (x-m)/s
    return exp(-0.5*z*z)/(sqrt(2.0*np.pi)*s)

# optimisation test functions, see https://en.wikipedia.org/wiki/Test_functions_for_optimization

cdef double log_eggbox(double x, double y):
    cdef double tmp = 2.0+cos(x/2.)*cos(y/2.)
    return -5.0*log(tmp)

cdef double ackley(double x, double y):
    cdef double r = sqrt(0.5*(x*x+y*y))
    cdef double first = 20.0*exp(r)
    cdef double second = exp(0.5*(cos(2.0*np.pi*x)+cos(2.0*np.pi*y)))
    return -(first+second-exp(1)-20)

cdef double camel(double x, double y):
    cdef double x2 = x*x
    cdef double x4 = x2*x2
    cdef double x6 = x4*x2
    return -(2.0*x2-1.05*x4+x6/6.0+x*y+y*y)

cdef double bukin(double x, double y):
    return -(100.0*sqrt(fabs(y-0.01*x*x))+0.01*fabs(x+10.0))

cdef double cross_in_tray(double x, double y):
    return -(0.0001*(fabs(sin(x)*sin(y)*exp(fabs(100.0-sqrt(x*x+y*y)/np.pi)))+1)**0.1)

cdef double rosenbrock(double x, double y):
    return -(100.0*(y-x*x)*(y-x*x)+(x-1)*(x-1))

cdef double rastrigin(double x, double y):
    return -(20.0+(x*x-10.0*cos(2.0*np.pi*x))+(y*y-10.0*cos(2.0*np.pi*y)))
