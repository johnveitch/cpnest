import sys
import os
import optparse as op
import numpy as np
from numpy.linalg import eig
from collections import deque
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue
from multiprocessing.managers import SyncManager

from . import parameter
from . import proposals

class Sampler(object):
    def __init__(self,usermodel,maxmcmc,verbose=True):
        self.user = usermodel
        self.data = usermodel.data
        names = usermodel.par_names
        bounds = usermodel.bounds
        self.cache = deque(maxlen=5*maxmcmc)
        self.maxmcmc = maxmcmc
        self.Nmcmc = maxmcmc
        self.proposals = proposals.setup_proposals_cycle()
        self.poolsize = 100
        self.evolution_points = [None]*self.poolsize
        self.verbose=verbose
        self.inParam = parameter.LivePoint(names,bounds)
        self.param = parameter.LivePoint(names,bounds)
        self.dimension = self.param.dimension
        for n in xrange(self.poolsize):
            while True:
                if self.verbose: sys.stderr.write("process {0!s} --> generating pool of {1:d} points for evolution --> {2:.3f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(n+1)/float(self.poolsize)))
                self.evolution_points[n] = parameter.LivePoint(names,bounds)
                self.evolution_points[n].logP = self.user.log_prior(self.evolution_points[n])
                if not(np.isinf(self.evolution_points[n].logP)): break
        if self.verbose: sys.stderr.write("\n")
        self.kwargs = proposals.ProposalArguments(self.dimension)
        self.kwargs.update(self.evolution_points)

    def produce_sample(self, consumer_lock, queue, IDcounter, logLmin, seed, ip, port, authkey):
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.counter=0

        while(1):
            IDcounter.get_lock().acquire()
            job_id = IDcounter.get_obj()
            id = job_id.value
            job_id.value+=1
            IDcounter.get_lock().release()
            if logLmin.value==999:
                break
            acceptance,jumps,outParam = self.metropolis_hastings(self.inParam,logLmin.value,self.Nmcmc)
            parameter.copy_live_point(self.evolution_points[np.random.randint(self.poolsize)],outParam)

            queue.put((id,acceptance,jumps,np.array([outParam.get(n) for n in outParam.names]),outParam.logP,outParam.logL))
            if self.counter == 0 and len(self.cache)==5*self.maxmcmc:
                self.autocorrelation()
                self.kwargs.update(self.evolution_points)
            elif (self.counter%(self.poolsize/4))==0:
                self.autocorrelation()
                self.kwargs.update(self.evolution_points)
            self.counter += 1
        sys.stderr.write("Sampler process {0!s}, exiting\n".format(os.getpid()))
        return 0

    def metropolis_hastings(self,inParam,logLmin,nsteps):
        """
        mcmc loop to generate the new live point taking nmcmc steps
        """
        accepted = 0
        rejected = 1
        jumps = 0
        parameter.copy_live_point(self.param,inParam)
        logp_old = self.user.log_prior(self.param)
        while (jumps < nsteps or accepted==0):
            self.param,log_acceptance = self.proposals[jumps%100].get_sample(self.param,self.evolution_points,self.kwargs)
            self.param.logP = self.user.log_prior(self.param)
            if self.param.logP-logp_old > log_acceptance:
                self.param.logL = self.user.log_likelihood(self.param)
                if self.param.logL > logLmin:
                    parameter.copy_live_point(inParam,self.param)
                    logp_old = self.param.logP
                    accepted+=1
                else:
                    parameter.copy_live_point(self.param,inParam)
                    rejected+=1
            else:
                parameter.copy_live_point(self.param,inParam)
                rejected+=1
                self.cache.append(np.array(([x.value for x in inParam.parameters])))
            jumps+=1
            if jumps==10*self.maxmcmc:
                return (0.0,jumps,inParam)
#            exit()
        return (float(accepted)/float(rejected+accepted),jumps,inParam)

    def autocorrelation(self):
        """
        estimates the autocorrelation length of the mcmc chain from the cached samples
        """
        try:
            ACLs = []
            cov_array = np.array(self.cache).T
            N =  cov_array.shape[1]
            for i,n in enumerate(self.evolution_points[0].names):
                ACF = proposals.autocorrelation(cov_array[i,:])
                ACL = np.min(np.where((ACF > -2./np.sqrt(N)) & (ACF < 2./np.sqrt(N)))[0])
                if not(np.isnan(ACL)):
                    ACLs.append(ACL)
                    if self.verbose: sys.stderr.write("process {0!s} --> cache size: {1:d} autocorrelation length {2!s} = {3:.1f} mean = {4:g} standard deviation = {5:g}\n".format(os.getpid(), len(self.cache), n, ACLs[-1], np.mean(cov_array[:,i]), np.std(cov_array[:,i])))
            self.Nmcmc =int((np.max(ACLs)))
            if self.Nmcmc < 2: self.Nmcmc = 2
            if self.Nmcmc > self.maxmcmc:
                sys.stderr.write("Warning ACL --> {0:d}!\n".format(self.Nmcmc))
                self.Nmcmc = self.maxmcmc
        except:
            sys.stderr.write("Warning ACL failed! setting {0:d}!\n".format(self.maxmcmc))
            self.Nmcmc = self.maxmcmc
        if len(self.cache)==5*self.maxmcmc:
            self.cache = deque(maxlen=5*self.maxmcmc)


