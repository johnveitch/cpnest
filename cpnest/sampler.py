from __future__ import division
import sys
import os
import logging
import numpy as np
from math import log
from collections import deque
from random import random,randrange

from . import parameter
from .proposal import DefaultProposalCycle, EnsembleProposal
from . import proposal
import numpy.lib.recfunctions as rfn
import array
from .nest2pos import acl, autocorrelation
import ray

import pickle
__checkpoint_flag = False


class Sampler(object):
    """
    Sampler class.
    ---------

    Initialisation arguments:

    args:
    model: :obj:`cpnest.Model` user defined model to sample

    maxmcmc:
        :int: maximum number of mcmc steps to be used in the :obj:`cnest.sampler.Sampler`

    ----------
    kwargs:

    verbose:
        :int: display debug information on screen
        Default: 0

    nlive:
        :int: number of live points. Used only for the nmcmc estimation on the fly

    proposal:
        :obj:`cpnest.proposals.Proposal` to use
        Defaults: :obj:`cpnest.proposals.DefaultProposalCycle`)
    """

    def __init__(self,
                 model,
                 maxmcmc,
                 verbose      = False,
                 nlive        = 1000,
                 proposal     = None):

        self.counter        = 0
        self.model          = model
        self.initial_mcmc   = maxmcmc//10
        self.maxmcmc        = maxmcmc
        self.tau            = None
        self.logger         = logging.getLogger('CPNest')

        if proposal is None:
            self.proposal = DefaultProposalCycle()
        else:
            self.proposal = proposal

        self.Nmcmc              = self.initial_mcmc
        self.Nmcmc_exact        = float(self.initial_mcmc)

        self.nlive              = nlive
        self.verbose            = verbose
        self.acceptance         = 0.0
        self.sub_acceptance     = 0.0
        self.mcmc_accepted      = 0
        self.mcmc_counter       = 0
        self.initialised        = False
        self.max_storage        = 32768
        self.samples            = deque(maxlen = None if self.verbose >=3 else self.max_storage) # the list of samples from the mcmc chain

    def estimate_nmcmc_on_the_fly(self, safety=5, tau=None):
        """
        Estimate autocorrelation length of chain using acceptance fraction
        ACL = (2/acc) - 1
        multiplied by a safety margin of 5
        Uses moving average with decay time tau iterations
        (default: :int:`self.nlive`)

        Taken from http://github.com/farr/EnsembleNest.jl
        """
        # if no auto correlation time is provided, make the safest assumption
        # that maxmcmc is safety times the auto correlation time
        if tau is None: tau  = self.maxmcmc

        # if we accepted no points, increase the number of steps
        if self.sub_acceptance == 0.0:
            self.Nmcmc_exact = (1.0 + 1.0/tau)*self.Nmcmc_exact
        else:
            self.Nmcmc_exact = (1.0 - 1.0/tau)*self.Nmcmc_exact + (safety/tau)*(2.0/self.sub_acceptance - 1.0)

        self.Nmcmc_exact = float(min(self.Nmcmc_exact,self.maxmcmc))
        self.Nmcmc = max(safety,int(self.Nmcmc_exact))

        return self.Nmcmc

    def estimate_acl(self, safety=1):
        """
        Estimate autocorrelation length of the chain
        
        safety determines the number of ACL the chains are going to be run for
        """
        # first of all, build a numpy array out of
        # the stored samples
        ACL = []
        s   = list(self.samples)
        samples = np.array([x.values for x in s[-self.max_storage:]])
        # compute the ACL on the max_storage set of samples, choosing a window length of 5*tau
        ACL = [acl(samples[:,i], c=5) for i in range(samples.shape[1])]
        
        if self.verbose >= 3:
            for i in range(len(self.model.names)):
                self.logger.info("Sampler {0} -- ACL({1})  = {2}".format(os.getpid(),self.model.names[i],ACL[i]))
        
        # take the longest ACL as the auto correlation time of the chain
        self.tau = np.max(ACL)
        # set the number of steps to be safety times the ACL
        self.Nmcmc = int(safety*self.tau)
        return self.Nmcmc

    def produce_sample(self, old, logLmin):
        """
        main loop that takes the worst :obj:`cpnest.parameter.LivePoint` and
        evolves it. Proposed sample is then sent back
        to :obj:`cpnest.NestedSampler`.
        """
        (Nmcmc, outParam) = self.return_sample(old, logLmin)
        # Send the sample to the Nested Sampler

        self.counter += 1
        if self.counter%10 == 0:
            self.estimate_acl()

        return self.acceptance, self.sub_acceptance, Nmcmc, outParam

    def set_ensemble(self, ensemble):
        if self.verbose > 3:
            self.logger.info("Sampler {0} -- setting ensemble".format(os.getpid()))

        for p in self.proposal.proposals:
            if isinstance(p, EnsembleProposal):
                p.set_ensemble(ensemble)
        return 0

@ray.remote
class MetropolisHastingsSampler(Sampler):
    """
    metropolis-hastings acceptance rule
    for :obj:`cpnest.proposal.EnembleProposal`
    """
    def return_sample(self, oldparam, logLmin):

        sub_counter = 0
        sub_accepted = 0
        logp_old = self.model.log_prior(oldparam)

        while True:

            sub_counter += 1
            newparam = self.proposal.get_sample(oldparam.copy())
            newparam.logP = self.model.log_prior(newparam)

            if newparam.logP-logp_old + self.proposal.log_J > log(random()):

                newparam.logL = self.model.log_likelihood(newparam)

                if newparam.logL > logLmin:

                    oldparam = newparam.copy()
                    logp_old = newparam.logP
                    sub_accepted+=1

            # append the sample to the array of samples
            self.samples.append(oldparam)

            if (sub_counter >= self.Nmcmc and sub_accepted > 0 ) or sub_counter >= self.maxmcmc:
                break

        self.sub_acceptance = float(sub_accepted)/float(sub_counter)
        self.mcmc_accepted += sub_accepted
        self.mcmc_counter += sub_counter
        self.acceptance    = float(self.mcmc_accepted)/float(self.mcmc_counter)
        # return the new sample
        return (sub_counter, oldparam)

@ray.remote
class HamiltonianMonteCarloSampler(Sampler):
    """
    HamiltonianMonteCarlo acceptance rule
    for :obj:`cpnest.proposal.HamiltonianProposal`
    """
    def return_sample(self, oldparam, logLmin):

        sub_accepted    = 0
        sub_counter     = 0

        while True:

            sub_counter += 1
            newparam     = self.proposal.get_sample(oldparam.copy(), logLmin = logLmin)

            if self.proposal.log_J > np.log(random()):

                if newparam.logL > logLmin:
                    oldparam        = newparam.copy()
                    sub_accepted   += 1

            # append the sample to the array of samples
            self.samples.append(oldparam)
        
            if sub_counter >= self.Nmcmc and sub_accepted > 0:
                break

            if sub_counter >= self.maxmcmc:
                break

        self.sub_acceptance = float(sub_accepted)/float(sub_counter)
        self.mcmc_accepted += sub_accepted
        self.mcmc_counter  += sub_counter
        self.acceptance     = float(self.mcmc_accepted)/float(self.mcmc_counter)

#            for p in self.proposal.proposals:
#                p.update_time_step(self.acceptance)
#                p.update_trajectory_length(safety=10)
        return (sub_counter, oldparam)

@ray.remote
class SliceSampler(Sampler):
    """
    The Ensemble Slice sampler from Karamanis & Beutler
    https://arxiv.org/pdf/2002.06212v1.pdf
    """
    mu             = 1.0
    max_steps_out  = 100 # maximum stepping out steps allowed
    max_slices     = 100 # maximum number of slices allowed
    
    def adapt_length_scale(self):
        """
        adapts the length scale of the expansion/contraction
        following the rule in (Robbins and Monro, 1951) of Tibbits et al. (2014)
        """
        Ne = max(1,self.Ne)
        Nc = max(1,self.Nc)
        ratio = Ne/(Ne+Nc)
        self.mu *= 2*ratio

    def reset_boundaries(self):
        """
        resets the boundaries and counts
        for the slicing
        """
        self.L  = - np.random.uniform(0.0,1.0)
        self.R  = self.L + 1.0
        self.Ne = 0.0
        self.Nc = 0.0

    def increase_left_boundary(self):
        """
        increase the left boundary and counts
        by one unit
        """
        self.L  = self.L - 1.0
        self.Ne = self.Ne + 1

    def increase_right_boundary(self):
        """
        increase the right boundary and counts
        by one unit
        """
        self.R  = self.R + 1.0
        self.Ne = self.Ne + 1

    def return_sample(self, oldparam, logLmin):

        sub_accepted    = 0
        sub_counter     = 0

        while True:
            # Set Initial Interval Boundaries
            self.reset_boundaries()
            sub_counter += 1

            direction_vector = self.proposal.get_direction(mu = self.mu)
            if not(isinstance(direction_vector,parameter.LivePoint)):
                direction_vector = parameter.LivePoint(oldparam.names,d=direction_vector)

            Y = logLmin
            Yp = oldparam.logP-np.random.exponential()
            J = np.floor(self.max_steps_out*np.random.uniform(0,1))
            K = (self.max_steps_out-1)-J
            # keep on expanding until we get outside the logL boundary from the left
            # or the prior bound, whichever comes first
            while J > 0:

                parameter_left = direction_vector * self.L + oldparam

                if self.model.in_bounds(parameter_left):
                    if Yp > self.model.log_prior(parameter_left):
                        break
                    else:
                        self.increase_left_boundary()
                        J -= 1
                # if we get out of bounds, break out
                else:
                    break

            # keep on expanding until we get outside the logL boundary from the right
            # or the prior bound, whichever comes first
            while K > 0:

                parameter_right = direction_vector * self.R + oldparam

                if self.model.in_bounds(parameter_right):

                    if Yp > self.model.log_prior(parameter_right):
                        break
                    else:
                        self.increase_right_boundary()
                        K -= 1
                # if we get out of bounds, break out
                else:
                    break

            # slice sample the likelihood-bound prior
            slice = 0
            while slice < self.max_slices:
                # generate a new point between the boundaries we identified
                Xprime        = np.random.uniform(self.L,self.R)
                newparam      = direction_vector * Xprime + oldparam
                newparam.logP = self.model.log_prior(newparam)

                if newparam.logP > Yp:
                    # compute the new value of logL
                    newparam.logL = self.model.log_likelihood(newparam)
                    if newparam.logL > Y:
                        oldparam     = newparam.copy()
                        sub_accepted += 1
                        break
                # adapt the intervals shrinking them
                if Xprime < 0.0:
                    self.L = Xprime
                    self.Nc = self.Nc + 1
                elif Xprime > 0.0:
                    self.R = Xprime
                    self.Nc = self.Nc + 1

                slice += 1

            if sub_counter >= self.Nmcmc and sub_accepted > 0:
                break

            if sub_counter >= self.maxmcmc:
                break

            self.samples.append(oldparam)
        self.sub_acceptance = float(sub_accepted)/float(sub_counter)
        self.mcmc_accepted += sub_accepted
        self.mcmc_counter  += sub_counter
        self.acceptance     = float(self.mcmc_accepted)/float(self.mcmc_counter)

        if logLmin == -np.inf:
            self.adapt_length_scale()

        return (sub_counter, oldparam)

@ray.remote
class SamplersCycle(Sampler):
    """
    A sampler that cycles through a list of
    samplers.

    Initialisation arguments:

    samplers : A list of samplers
    weights   : Weights for each type of jump

    Optional arguments:
    cyclelength : length of the proposal cycle. Default: 100

    """
    idx = 0 # index in the cycle
    N   = 0   # number of samplers in the cycle
    def __init__(self,samplers,*args,**kwargs):
        self.samplers = samplers
        self.N        = len(self.samplers)

    def produce_sample(self, old, logLmin, **kwargs):
        # Call the current sampler and increment the index
        self.idx = (self.idx + 1) % self.N
        p = self.samplers[self.idx]
        acceptance, sub_acceptance, Nmcmc, new = p.produce_sample(old, logLmin, **kwargs)
        return acceptance, sub_acceptance, Nmcmc, new

    def set_ensemble(self, ensemble):
        """
        Updates the ensemble statistics
        by calling it on each :obj:`EnsembleProposal`
        """
        self.ensemble = ensemble
        for s in self.samplers:
            for p in s.proposal.proposals:
                if isinstance(p, EnsembleProposal):
                    p.set_ensemble(self.ensemble)
