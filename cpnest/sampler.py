import sys
import os
import numpy as np
from math import log
from collections import deque
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue
from random import random,randrange

from . import parameter
from . import proposal

class Sampler(object):
    """
    Sampler class.
    Initialisation arguments:
    
    usermodel:
    user defined model to sample
    
    maxmcmc:
    maximum number of mcmc steps to be used in the sampler
    default: 4096
    
    verbose:
    display debug information on screen
    default: False
    
    poolsize:
    number of objects for the affine invariant sampling
    default: 100
    
    """
    def __init__(self,usermodel,maxmcmc,verbose=False,poolsize=100):
        self.user = usermodel
        self.maxmcmc = maxmcmc
        self.Nmcmc = maxmcmc
        self.Nmcmc_exact = float(maxmcmc)
        self.proposals = proposal.DefaultProposalCycle()
        self.poolsize = poolsize
        self.evolution_points = deque(maxlen=self.poolsize)
        self.verbose=verbose
        self.inParam = self.user.new_point()
        self.dimension = self.inParam.dimension
        self.acceptance=0.0
        self.initialised=False
        
    def reset(self):
        for n in range(self.poolsize):
          while True:
            if self.verbose: sys.stderr.write("process {0!s} --> generating pool of {1:d} points for evolution --> {2:.0f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(n+1)/float(self.poolsize)))
            p = self.user.new_point()
            p.logP = self.user.log_prior(p)
            if np.isfinite(p.logP): break
          p.logL=self.user.log_likelihood(p)
          self.evolution_points.append(p)
        if self.verbose: sys.stderr.write("\n")
        self.proposals.set_ensemble(self.evolution_points)
        for _ in range(len(self.evolution_points)):
          s = self.evolution_points.popleft()
          acceptance,jumps,s = self.metropolis_hastings(s,-np.inf,self.Nmcmc)
          self.evolution_points.append(s)
        self.proposals.set_ensemble(self.evolution_points)
        self.initialised=True

    def estimate_nmcmc(self, safety=5, tau=None):
        """
        Estimate autocorrelation length of chain using acceptance fraction
        ACL = (2/acc) - 1
        multiplied by a safety margin of 5
        Uses moving average with decay time tau iterations (default: self.poolsize)
        Taken from W. Farr's github.com/farr/Ensemble.jl
        """
        if tau is None: tau = self.poolsize
        if self.acceptance==0:
            self.Nmcmc_exact = (1.0 + 1.0/tau)*self.Nmcmc_exact
        else:
            self.Nmcmc_exact = (1.0 - 1.0/tau)*self.Nmcmc_exact + (safety/tau)*(2.0/self.acceptance - 1.0)
        self.Nmcmc = max(safety,min(self.maxmcmc, int(round(self.Nmcmc_exact))))
        return self.Nmcmc

    def produce_sample(self, consumer_lock, queue, IDcounter, logLmin, seed, ip, port, authkey):
        """
        main loop that generates samples and puts them in the queue for the nested sampler object
        """
        if not self.initialised:
          self.reset()
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.counter=0
        while(1):
            # Pick a random point from the ensemble to start with
            # Pop it out the stack to prevent cloning
            self.inParam = self.evolution_points.popleft()
            IDcounter.get_lock().acquire()
            job_id = IDcounter.get_obj()
            id = job_id.value
            job_id.value+=1
            IDcounter.get_lock().release()
            if logLmin.value==np.inf:
                break
            acceptance,jumps,outParam = self.metropolis_hastings(self.inParam,logLmin.value,self.Nmcmc)
            # If we bailed out then flag point as unusable
            if acceptance==0.0:
                outParam.logL=-np.inf
            # Put sample back in the stack
            self.evolution_points.append(outParam.copy())
            queue.put((id,acceptance,jumps,outParam))
            if (self.counter%(self.poolsize/10))==0 or self.acceptance<0.01:
                #self.autocorrelation()
                self.proposals.set_ensemble(self.evolution_points)
            self.counter += 1
        sys.stderr.write("Sampler process {0!s}, exiting\n".format(os.getpid()))
        return 0

    def metropolis_hastings(self,inParam,logLmin,nsteps):
        """
        metropolis-hastings loop to generate the new live point taking nmcmc steps
        """
        accepted = 0
        rejected = 0
        jumps = 0
        oldparam=inParam
        logp_old = self.user.log_prior(oldparam)
        while (jumps < nsteps):
            newparam = self.proposals.get_sample(oldparam.copy())
            newparam.logP = self.user.log_prior(newparam)
            if newparam.logP-logp_old + self.proposals.log_J > log(random()):
                newparam.logL = self.user.log_likelihood(newparam)
                if newparam.logL > logLmin:
                  oldparam=newparam
                  logp_old = newparam.logP
                  accepted+=1
                else:
                  rejected+=1
            else:
                rejected+=1

            jumps+=1
            if jumps==10*self.maxmcmc:
              print('Warning, MCMC chain exceeded {0} iterations!'.format(10*self.maxmcmc))
        self.acceptance = float(accepted)/float(jumps)
        self.estimate_nmcmc()
        return (float(accepted)/float(rejected+accepted),jumps,oldparam)


if __name__=="__main__":
    pass
