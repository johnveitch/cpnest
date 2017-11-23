from __future__ import division
import sys
import os
import numpy as np
from math import log
from collections import deque
from random import random,randrange

from . import parameter
from .proposal import DefaultProposalCycle

class Sampler(object):
    """
    Sampler class.
    Initialisation arguments:
    
    usermodel:
    user defined model to sample
    
    maxmcmc:
    maximum number of mcmc steps to be used in the sampler
    
    verbose:
    display debug information on screen
    default: False
    
    poolsize:
    number of objects for the affine invariant sampling
    default: 1000
    
    seed:
    Random seed
    
    proposal:
    JumpProposal class to use (defaults to proposals.DefaultProposalCycle)
    
    """
    def __init__(self,usermodel,maxmcmc,seed=None, verbose=False, poolsize=1000, proposal=None):
        self.user = usermodel
        self.initial_mcmc = maxmcmc//2
        self.maxmcmc = maxmcmc

        if proposal is None:
            self.proposal = DefaultProposalCycle()
        else:
            self.proposal = proposal

        self.Nmcmc = self.initial_mcmc
        self.Nmcmc_exact = float(self.initial_mcmc)

        self.poolsize = poolsize
        self.evolution_points = deque(maxlen=self.poolsize)
        self.verbose=verbose
        self.acceptance=0.0
        self.initialised=False
        self.samples = [] # the list of samples from the mcmc chain
        
    def reset(self):
        """
        Initialise the sampler
        """
        np.random.seed(seed=self.seed)
        for n in range(self.poolsize):
          while True:
            if self.verbose > 2: sys.stderr.write("process {0!s} --> generating pool of {1:d} points for evolution --> {2:.0f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(n+1)/float(self.poolsize)))
            p = self.user.new_point()
            p.logP = self.user.log_prior(p)
            if np.isfinite(p.logP): break
          p.logL=self.user.log_likelihood(p)
          self.evolution_points.append(p)
        if self.verbose > 2: sys.stderr.write("\n")

        self.proposals.set_ensemble(self.evolution_points)

        self.metropolis_hastings(-np.inf)
        
        if self.verbose > 2: sys.stderr.write("\n")
        if self.verbose > 2: sys.stderr.write("Initial estimated ACL = {0:d}\n".format(self.Nmcmc))
        self.proposals.set_ensemble(self.evolution_points)

        self.initialised=True

    def estimate_nmcmc(self, safety=1, tau=None):
        """
        Estimate autocorrelation length of chain using acceptance fraction
        ACL = (2/acc) - 1
        multiplied by a safety margin of 5
        Uses moving average with decay time tau iterations (default: self.poolsize)
        Taken from W. Farr's github.com/farr/Ensemble.jl
        """
        if tau is None: tau = self.poolsize

        if self.acceptance == 0.0:
            self.Nmcmc_exact = (1.0 + 1.0/tau)*self.Nmcmc_exact
        else:
            self.Nmcmc_exact = (1.0 - 1.0/tau)*self.Nmcmc_exact + (safety/tau)*(2.0/self.acceptance - 1.0)
        
        self.Nmcmc_exact = float(min(self.Nmcmc_exact,self.maxmcmc))
        self.Nmcmc = max(safety,int(self.Nmcmc_exact))

        return self.Nmcmc

    def produce_sample(self, queue, logLmin ):
        """
        main loop that generates samples and puts them in the queue for the nested sampler object
        """
        if not self.initialised:
          self.reset()
        # Prevent process from zombification if consumer thread exits
        queue.cancel_join_thread()
        
        self.counter=0
        
        while True:
            if logLmin.value==np.inf:
                break

            self.metropolis_hastings(logLmin.value)
            outParam = self.evolution_points[np.random.randint(self.poolsize)]
            # Push the sample onto the queue
            queue.put((self.acceptance,self.Nmcmc,outParam))
            # Update the ensemble every now and again
            if (self.counter%(self.poolsize/10))==0 or self.acceptance<1.0/float(self.poolsize):
                self.proposals.set_ensemble(self.evolution_points)
            self.counter += 1

        print "MCMC samples accumulated = ",len(self.samples)
        import numpy.lib.recfunctions as rfn
        self.mcmc_samples = rfn.stack_arrays([self.samples[j].asnparray() for j in range(len(self.samples))],usemask=False)
        np.savetxt(os.path.join('mcmc_chain_%s.dat'%os.getpid()),self.mcmc_samples.ravel(),header=' '.join(self.mcmc_samples.dtype.names),newline='\n',delimiter=' ')
        sys.stderr.write("Sampler process {0!s}, saved chain in {1!s}\n".format(os.getpid(),'mcmc_chain_%s.dat'%os.getpid()))
        sys.stderr.write("Sampler process {0!s}, exiting\n".format(os.getpid()))
        return 0

    def metropolis_hastings(self, logLmin):
        """
        metropolis-hastings loop to generate the new live point taking nmcmc steps
        """
        accepted = 0

        for j in range(self.poolsize):

            oldparam = self.evolution_points.popleft()
            logp_old = self.user.log_prior(oldparam)

            for n in range(self.Nmcmc):

                newparam = self.proposals.get_sample(oldparam.copy())
                newparam.logP = self.user.log_prior(newparam)
                if newparam.logP-logp_old + self.proposals.log_J > log(random()):
                    newparam.logL = self.user.log_likelihood(newparam)
                    if newparam.logL > logLmin:
                      oldparam = newparam
                      logp_old = newparam.logP
                      accepted+=1
            # Put sample back in the stack
            self.samples.append(oldparam)
            self.evolution_points.append(oldparam)

        self.acceptance = float(accepted)/float(self.poolsize*self.Nmcmc)
        self.estimate_nmcmc()
