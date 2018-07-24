from __future__ import division
import sys
import os
import numpy as np
from math import log
from collections import deque
from random import random,randrange

from . import parameter
from .proposal import DefaultProposalCycle
from . import proposal

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

    def __init__(self,usermodel, maxmcmc, seed=None, output=None, verbose=False, poolsize=1000, proposal=None):

        self.seed = seed
        self.user = usermodel
        self.initial_mcmc = maxmcmc//10
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
        self.sub_acceptance=0.0
        self.mcmc_accepted = 0
        self.mcmc_counter = 0
        self.initialised=False
        self.output = output
        self.samples = [] # the list of samples from the mcmc chain
        self.ACLs = [] # the history of the ACL of the chain, will be used to thin the output, if requested
        
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
          if p.logL is None or not np.isfinite(p.logL):
              print("Warning: received non-finite logL value {0} with parameters {1}".format(str(p.logL), str(p)))
              print("You may want to check your likelihood function to impove sampling")
          self.evolution_points.append(p)
        if self.verbose > 2: sys.stderr.write("\n")

        self.proposal.set_ensemble(self.evolution_points)

        self.metropolis_hastings(-np.inf)
        
        if self.verbose > 2: sys.stderr.write("\n")
        if self.verbose > 2: sys.stderr.write("Initial estimated ACL = {0:d}\n".format(self.Nmcmc))
        self.proposal.set_ensemble(self.evolution_points)

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

        if self.sub_acceptance == 0.0:
            self.Nmcmc_exact = (1.0 + 1.0/tau)*self.Nmcmc_exact
        else:
            self.Nmcmc_exact = (1.0 - 1.0/tau)*self.Nmcmc_exact + (safety/tau)*(2.0/self.sub_acceptance - 1.0)
        
        self.Nmcmc_exact = float(min(self.Nmcmc_exact,self.maxmcmc))
        self.Nmcmc = max(safety,int(self.Nmcmc_exact))
        self.ACLs.append(self.Nmcmc)

        return self.Nmcmc

    def produce_sample(self, producer_pipe, logLmin ):
        """
        main loop that generates samples and puts them in the queue for the nested sampler object
        """

        if not self.initialised:
          self.reset()

        self.counter=0
        
        while True:
            if logLmin.value==np.inf:
                break

            p = producer_pipe.recv()

            if p is None:
                break
            
            self.evolution_points.append(p)
            (acceptance,Nmcmc,outParam) = next(self.metropolis_hastings(logLmin.value))

            # Send the sample to the Nested Sampler
            producer_pipe.send((acceptance,Nmcmc,outParam))
            # Update the ensemble every now and again

            if (self.counter%(self.poolsize/10))==0 or acceptance < 1.0/float(self.poolsize):
                self.proposal.set_ensemble(self.evolution_points)
            self.counter += 1

        sys.stderr.write("Sampler process {0!s}: MCMC samples accumulated = {1:d}\n".format(os.getpid(),len(self.samples)))
        thinning = int(np.ceil(np.mean(self.ACLs)))
        self.samples.extend(self.evolution_points)
        sys.stderr.write("Sampler process {0!s}: Mean ACL measured (suggested thinning) = {1:d}\n".format(os.getpid(),thinning))
        import numpy.lib.recfunctions as rfn
        self.mcmc_samples = rfn.stack_arrays([self.samples[j].asnparray() for j in range(0,len(self.samples))],usemask=False)
        if self.verbose >=3:
            np.savetxt(os.path.join(self.output,'mcmc_chain_%s.dat'%os.getpid()),
                       self.mcmc_samples.ravel(),header=' '.join(self.mcmc_samples.dtype.names),
                       newline='\n',delimiter=' ')
            sys.stderr.write("Sampler process {0!s}: saved {1:d} mcmc samples in {2!s}\n".format(os.getpid(),len(self.samples),'mcmc_chain_%s.dat'%os.getpid()))
        sys.stderr.write("Sampler process {0!s} - mean acceptance {1:.3f}: exiting\n".format(os.getpid(), float(self.mcmc_accepted)/float(self.mcmc_counter)))
        return 0

    def metropolis_hastings(self, logLmin):
        """
        metropolis-hastings loop to generate the new live point taking nmcmc steps
        """
        for j in range(self.poolsize):
            sub_counter = 0
            sub_accepted = 0
            oldparam = self.evolution_points.popleft()
            logp_old = self.user.log_prior(oldparam)

            while True:
                sub_counter += 1
                newparam = self.proposal.get_sample(oldparam.copy())
                newparam.logP = self.user.log_prior(newparam)
                if newparam.logP-logp_old + self.proposal.log_J > log(random()):
                    newparam.logL = self.user.log_likelihood(newparam)
                    if newparam.logL > logLmin:
                        oldparam = newparam
                        logp_old = newparam.logP
                        sub_accepted+=1
                if (sub_counter > self.Nmcmc and sub_accepted > 0 ) or sub_counter > self.maxmcmc:
                    break
        
            # Put sample back in the stack
            self.evolution_points.append(oldparam)
            if self.verbose >=3: self.samples.append(oldparam)
            self.sub_acceptance = float(sub_accepted)/float(sub_counter)
            self.estimate_nmcmc()
            self.mcmc_accepted += sub_accepted
            self.mcmc_counter += sub_counter
            # Yield the new sample
            if oldparam.logL > logLmin:
                yield (float(self.sub_acceptance),sub_counter,oldparam)
