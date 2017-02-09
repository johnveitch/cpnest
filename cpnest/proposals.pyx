# encoding: utf-8
# cython: profile=True
# cython: linetrace=True

from __future__ import division
from functools import reduce
import numpy as np
cimport numpy as np
from libc.math cimport log,exp,sqrt,fabs

from . cimport parameter
from parameter cimport LivePoint

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cdef class ProposalArguments(object):
    cdef public unsigned int n
    cdef public unsigned int dimension
    cdef public np.ndarray eigen_values
    cdef public np.ndarray eigen_vectors
    cdef public list pool
    def __cinit__(self, unsigned int dimension):
        self.dimension = dimension
        self.eigen_values = np.zeros(self.dimension, dtype=DTYPE)
        self.eigen_vectors = np.zeros((self.dimension,self.dimension), dtype=DTYPE)

    def update(self, list pool):
        cdef unsigned int i,j,p
        cdef unsigned int n = len(pool)
        cdef np.ndarray[np.double_t, ndim=2, mode = 'c'] cov_array = np.zeros((self.dimension,n))
        cdef np.ndarray[DTYPE_t, ndim=2, mode = 'c'] covariance
        cdef str name
        if self.dimension == 1:
            name=pool[0].names[i]
            self.eigen_values = np.atleast_1d(np.var([pool[j][name] for j in range(n)]))
            self.eigen_vectors = np.eye(1)
        else:	 
            for i in range(self.dimension):
                name=pool[0].names[i]
                for j in range(n): cov_array[i,j] = pool[j][name]
            covariance = np.cov(cov_array)
            self.eigen_values,self.eigen_vectors = np.linalg.eigh(covariance)

cdef tuple _EnsembleEigenDirection(LivePoint inParam, list Ensemble, ProposalArguments arguments):
    cdef unsigned int i = np.random.randint(arguments.dimension)
    cdef unsigned int k
    cdef double jumpsize,log_acceptance_probability
    cdef LivePoint outParam = inParam
    for k in range(arguments.dimension):
        jumpsize = sqrt(fabs(arguments.eigen_values[i]))*np.random.normal(0,1)
        outParam[outParam.names[k]]+=jumpsize*arguments.eigen_vectors[k,i]
    log_acceptance_probability = log(np.random.uniform(0.,1.))
    return outParam,log_acceptance_probability

cdef tuple _EnsembleWalk(LivePoint inParam, list Ensemble, ProposalArguments arguments):
    cdef unsigned int i,j
    cdef unsigned int Nsubset = 3
    cdef np.ndarray[np.int64_t, ndim=1] indeces = np.random.choice(range(len(Ensemble)),Nsubset)
    cdef list subset = [Ensemble[i] for i in indeces]
    cdef LivePoint center_of_mass=reduce(LivePoint.__add__,subset)/float(Nsubset)
    cdef LivePoint outParam = inParam
    
    for j in range(Nsubset):
      outParam += (center_of_mass - subset[j])*np.random.normal(0,1)

    cdef double log_acceptance_probability = log(np.random.uniform(0.,1.))
    return outParam,log_acceptance_probability


cdef tuple _EnsembleStretch(LivePoint inParam, list Ensemble, ProposalArguments arguments):

    cdef unsigned int i,j
    cdef unsigned int dimension = arguments.dimension
    cdef unsigned int N = len(Ensemble)
    cdef double tmp
    cdef double scale =2.0
    cdef np.ndarray[DTYPE_t, ndim=1] wa = np.zeros(dimension, dtype=DTYPE)
    cdef int a = np.random.randint(N)
    cdef double log_acceptance_probability
    cdef double u = np.random.uniform(0.0,1.0)
    cdef double x = (2.0*u-1)*log(scale)
    cdef double Z = exp(x)
    cdef LivePoint outParam = Ensemble[a] + (inParam - Ensemble[a])*Z

    if (Z<1.0/scale)or(Z>scale):
        log_acceptance_probability = -np.inf
    else:
        log_acceptance_probability = log(np.random.uniform(0.,1.))-(dimension)*log(Z)
    return outParam,log_acceptance_probability

cdef tuple _DifferentialEvolution(LivePoint inParam, list Ensemble, ProposalArguments arguments):
    cdef unsigned int i
    cdef unsigned int dimension = arguments.dimension
    cdef double gamma = 2.38/sqrt(2.0*dimension)
    cdef np.ndarray[np.int64_t, ndim=1] indeces = np.random.choice(range(len(Ensemble)),2)
    cdef double mutation_variance = 1e-4

    cdef LivePoint outParam = inParam+(Ensemble[indeces[0]]-Ensemble[indeces[1]])*gamma
    
    for i in range(dimension):
        outParam.values[i] += mutation_variance*np.random.normal(0,1)

    cdef double log_acceptance_probability = log(np.random.uniform(0.,1.))
    return outParam,log_acceptance_probability


proposals = {}
proposal_list_name = ['DifferentialEvolution','EnsembleEigenDirections','EnsembleWalk','EnsembleStretch']
proposal_list = [_DifferentialEvolution,_EnsembleEigenDirection,_EnsembleWalk,_EnsembleStretch]

for name,algorithm in zip(proposal_list_name,proposal_list):
    proposals[name]=algorithm

cdef class Proposal(object):
    cdef public object algorithm
    cdef public str name
    
    def __cinit__(self, str name):

        self.name = name
        self.algorithm = proposals[name]

    cpdef get_sample(self, object inParam, list Ensemble,ProposalArguments Arguments):
        return self.algorithm(inParam,Ensemble,Arguments)

cpdef list setup_proposals_cycle():
    cdef unsigned int i
    cdef np.ndarray jump_proposals_names = np.random.choice(proposal_list_name,size = 100)
    return [Proposal(str(n)) for n in jump_proposals_names]

cpdef np.ndarray[DTYPE_t, ndim=1] autocorrelation(np.ndarray[DTYPE_t, ndim=1] x):
    #x = np.asarray(x)
    cdef unsigned int N = len(x)
    x -= x.mean()
    cdef np.ndarray s = np.fft.fft(x, N*2-1)
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.real(np.fft.ifft(s * np.conjugate(s), N*2-1))
    return result[:N]/result[0]
