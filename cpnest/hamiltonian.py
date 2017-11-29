from __future__ import division
import sys
import os
import numpy as np
from math import log
from collections import deque
from random import random,randrange

from . import parameter
from . import proposal
"""
hamilton equations collisional gas in a box whose side is shrinking
"""

class HMCSampler(object):
    """
    Hamiltonian Monte Carlo Sampler class.
    Initialisation arguments:
    
    usermodel:
    user defined model to sample
    
    maxmcmc:
    maximum number of mcmc steps to be used in the sampler
    
    verbose:
    display debug information on screen
    default: False
    
    poolsize:
    number of objects for the gradients estimation
    default: 1000
    """
    def __init__(self,usermodel, maxmcmc, verbose=False, poolsize=1000, l=1.0):
        self.user           = usermodel
        self.maxmcmc        = maxmcmc
        self.Nmcmc          = maxmcmc
        self.Nmcmc_exact    = float(maxmcmc)
        self.proposals      = proposal.DefaultProposalCycle()
        self.poolsize       = poolsize
        self.positions      = deque(maxlen=self.poolsize + 1) # +1 for the point to evolve
        self.verbose        = verbose
        self.acceptance     = 0.0
        self.initialised    = False
        self.gradients      = {}
        # step size choice from http://www.homepages.ucl.ac.uk/~ucakabe/papers/Bernoulli_11b.pdf
        # which asks for the relation step_size = l * dims**(1/4)
        self.l              = l
        self.step_size      = 1.0
        self.steps          = 20
        self.momenta_distribution = None
        self.counter = 0
    
    def reset(self):
        """
        Initialise the sampler
        """
        for n in range(self.poolsize):
            while True:
                if self.verbose > 2: sys.stderr.write("process {0!s} --> generating pool of {1:d} points for evolution --> {2:.0f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(n+1)/float(self.poolsize)))
                p = self.user.new_point()
                p.logP = self.user.log_prior(p)
                if np.isfinite(p.logP): break
            p.logL=self.user.log_likelihood(p)
            self.positions.append(p)
        if self.verbose > 2: sys.stderr.write("\n")

        self.proposals.set_ensemble(self.positions)

        # seed the chains with a standard MCMC
        for j in range(len(self.positions)):
            if self.verbose > 2: sys.stderr.write("process {0!s} --> initial MCMC evolution for {1:d} points --> {2:.0f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(j+1)/float(self.poolsize)))
            s = self.positions.popleft()
            s = self.metropolis_hastings(s,-np.inf)
            self.positions.append(s)
        if self.verbose > 2: sys.stderr.write("\n")

        self.proposals.set_ensemble(self.positions)
        self.momenta_distribution = multivariate_normal(cov=self.proposals.mass_matrix)
        self.step_size = self.l * float(len(self.positions[0].names))**(0.25)

        # estimate the initial gradients
        if self.verbose > 2: sys.stderr.write("Computing initial gradients ...")
        logProbs = np.array([-p.logP for p in self.positions])
        self.estimate_gradient(logProbs, 'logprior')
        logProbs = np.array([-p.logL for p in self.positions])
        self.estimate_gradient(logProbs, 'loglikelihood')

        if self.verbose > 2: sys.stderr.write("done\n")
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
#            print self.Nmcmc_exact,(1.0 - 1.0/tau)
        else:
            self.Nmcmc_exact = (1.0 - 1.0/tau)*self.Nmcmc_exact + (safety/tau)*(2.0/self.acceptance - 1.0)
#            print self.Nmcmc_exact,(1.0 - 1.0/tau),(safety/tau),(2.0/self.acceptance - 1.0),(safety/tau)*(2.0/self.acceptance - 1.0)

        self.Nmcmc_exact = float(min(self.Nmcmc_exact,self.maxmcmc))
        self.Nmcmc       = max(safety,int(self.Nmcmc_exact))

        return self.Nmcmc

    def produce_sample(self, queue, logLmin, seed, ip, port, authkey):
        """
        main loop that generates samples and puts them in the queue for the nested sampler object
        """
        if not self.initialised:
            self.reset()
        # Prevent process from zombification if consumer thread exits
        queue.cancel_join_thread()
        self.seed = seed
        np.random.seed(seed=self.seed)
        unused = 0
        while(1):
            if logLmin.value==np.inf:
                break
            # Pick a point from the ensemble to start with
            position = self.positions.popleft()
            # evolve it according to hamilton equations
            newposition = self.hamiltonian_sampling(position,logLmin.value)
            # Put sample back in the stack
            self.positions.append(newposition.copy())

            # If we bailed out then flag point as unusable
            if self.acceptance==0.0:
                newposition.logL=-np.inf
            # Push the sample onto the queue
            queue.put((self.acceptance,self.jumps,newposition))
            
            # Update the ensemble every now and again
            if (self.counter%(self.poolsize/10))==0:
                self.proposals.set_ensemble(self.positions)
                # update the gradients
                logProbs = np.array([-p.logP for p in self.positions])
                self.estimate_gradient(logProbs, 'logprior')
                logProbs = np.array([-p.logL for p in self.positions])
                self.estimate_gradient(logProbs, 'loglikelihood')
                self.momenta_distribution = multivariate_normal(cov=self.proposals.mass_matrix)

            self.counter += 1
        sys.stderr.write("Sampler process {0!s}, exiting\n".format(os.getpid()))
        return 0
    
    def estimate_gradient(self, logProbs, type):
        
        self.gradients[type] = []

        # loop over the parameters, estimate the gradient numerically and spline interpolate

#        import matplotlib.pyplot as plt

        for j,key in enumerate(self.positions[0].names):

            x = np.array([self.positions[i][key] for i in range(len(self.positions))])
            idx = np.argsort(x)
            lp = logProbs
            # let's use numpy gradient to compute the finite difference partial derivative of logProbs
            grad = np.gradient(lp,x)
            # check for nans
            w = np.isnan(grad)
            # zero the nans
            grad[w] = 0.
            # approximate with a rolling median and standard deviation to average out all the small scale variations
            window_size = 100
            N = np.maximum(np.ceil(self.poolsize/window_size).astype(int),3)
            bins = np.linspace(x.min(), x.max(),N)

            idx  = np.digitize(x,bins)
            running_median = np.array([np.nanmedian(grad[idx==k]) for k in range(N)])
            running_std    = np.array([np.nanstd(grad[idx==k]) for k in range(N)])
            weight = ~np.logical_or(np.isnan(running_median),np.isnan(running_std))
            running_median[~weight] = 0.0
            
            if not(np.any(np.isnan(running_std))):
                weight = weight.astype(float)/running_std
            
            # use now a linear spline interpolant to represent the partial derivative
            self.gradients[type].append(UnivariateSpline(bins, running_median, ext=0, k=3, w=weight, s=self.poolsize))
#            plt.figure()
#            plt.plot(x,lp,'ro',alpha=0.5)
#            plt.plot(x,grad,'g.',alpha=0.5)
#            plt.errorbar(bins,running_median,yerr=running_std,lw=2)
#            plt.plot(bins,self.gradients[type][j](bins),'k',lw=3)
#            plt.xlabel(key)
#            plt.ylim([-20,100])
#            plt.savefig('grad_%s_%s_%04d.png'%(type,key,self.counter))
#        plt.close('all')
#        exit()

    def gradient(self, inParam, gradients_list):
        return np.array([g(inParam[n]) for g,n in zip(gradients_list,self.positions[0].names)])

    def kinetic_energy(self, momentum):
        """Kinetic energy of the current velocity (assuming a standard Gaussian)
            (x dot x) / 2
        Parameters
        ----------
        velocity : tf.Variable
            Vector of current velocity
        Returns
        -------
        kinetic_energy : float
        """
        p = momentum.asnparray()
        p = p.view(dtype=np.float64)[:-2]

        return 0.5 * np.dot(p,np.dot(self.proposals.inverse_mass_matrix,p))

    def potential_energy(self, position):
        """
        potential energy of the current position
        """
        position.logP = self.user.log_prior(position)
        return -position.logP

    def hamiltonian(self, position, momentum):
        """Computes the Hamiltonian of the current position, velocity pair
        H = U(x) + K(v)
        U is the potential energy and is = -log_prior(x)
        Parameters
        ----------
        position : tf.Variable
            Position or state vector x (sample from the target distribution)
        velocity : tf.Variable
            Auxiliary velocity variable
        energy_function
            Function from state to position to 'energy'
             = -log_prior
        Returns
        -------
        hamitonian : float
        """
        return self.potential_energy(position) + self.kinetic_energy(momentum) #+position.logL

    def constrained_leapfrog_step(self, position, momentum, logLmin):
        """
        https://arxiv.org/pdf/1005.0157.pdf
        https://arxiv.org/pdf/1206.1901.pdf
        """
        # Updating the momentum a half-step
        g = self.gradient(position, self.gradients['logprior'])

        for j,k in enumerate(self.positions[0].names):
            momentum[k] += - 0.5 * self.step_size * g[j]

        for i in xrange(self.steps):
            
            # do a step
            for j,k in enumerate(self.positions[0].names):
                position[k] += self.step_size * momentum[k] * self.proposals.inverse_mass_matrix[j,j]
        
            # if the trajectory brings us outside the prior boundary, bounce back and forth
            # see https://arxiv.org/pdf/1206.1901.pdf pag. 37
            
            position.logP = self.user.log_prior(position)
            
            if not(np.isfinite(position.logP)):
                return position, momentum
                for j,k in enumerate(self.positions[0].names):
                    while position[k] > self.user.bounds[j][1]:
                        position[k] = self.user.bounds[j][1] - (position[k] - self.user.bounds[j][1])
                        momentum[k] = -momentum[k]
                    while position[k] < self.user.bounds[j][0]:
                        position[k] = self.user.bounds[j][0] + (self.user.bounds[j][0] - position[k])
                        momentum[k] = -momentum[k]

            # Update gradient
            g = self.gradient(position, self.gradients['logprior'])

            # compute the constraint
            position.logL = self.user.log_likelihood(position)

            # check on the constraint
            if position.logL > logLmin:
                # take a full momentum step
                for j,k in enumerate(self.positions[0].names):
                    momentum[k] += - self.step_size * g[j]
            else:
                # compute the normal to the constraint
                gL = self.gradient(position, self.gradients['loglikelihood'])
                n = gL/np.abs(np.sum(gL))
                
                # bounce on the constraint
                for j,k in enumerate(self.positions[0].names):
                    momentum[k] += - 2 * (momentum[k]*n[j]) * n[j]

#        # Update the position
#        for j,k in enumerate(self.positions[0].names):
#            position[k] += self.step_size * momentum[k]

        # Do a final update of the momentum for a half step
        g = self.gradient(position, self.gradients['logprior'])
        for j,k in enumerate(self.positions[0].names):
            momentum[k] += - 0.5 * self.step_size * g[j]

        return position, momentum
    
    def hamiltonian_sampling(self, initial_position, logLmin):
        """
        hamiltonian sampling loop to generate the new live point taking nmcmc steps
        """
        self.jumps  = 0
        accepted    = 0
        oldparam    = initial_position.copy()
        
        while self.jumps < self.Nmcmc:
            
            # generate the initial momentum from its canonical distribution
            v                = np.atleast_1d(self.momenta_distribution.rvs())
            initial_momentum = self.user.new_point()
            
            for j,k in enumerate(self.positions[0].names):
                initial_momentum[k] = v[j]

            starting_energy       = self.hamiltonian(oldparam, initial_momentum)
            newparam, newmomentum = self.constrained_leapfrog_step(oldparam.copy(), initial_momentum.copy(), logLmin)
            current_energy        = self.hamiltonian(newparam, newmomentum)

            logp_accept = min(0.0, starting_energy - current_energy)
    
            if logp_accept > np.log(random()):
                if newparam.logL > logLmin:
                    oldparam        = newparam
                    accepted       += 1
            
            self.jumps+=1
            if self.jumps > self.maxmcmc: break

        self.acceptance = float(accepted)/float(self.jumps)
        self.estimate_nmcmc()
#        self.autotune()
        return oldparam

    def metropolis_hastings(self, inParam, logLmin):
        """
        metropolis-hastings loop to generate the new live point taking nmcmc steps
        """
        self.jumps  = 0
        accepted    = 0
        oldparam    = inParam.copy()
        logp_old    = self.user.log_prior(oldparam)
        
        while self.jumps < self.Nmcmc:
            
            newparam        = self.proposals.get_sample(oldparam.copy())
            newparam.logP   = self.user.log_prior(newparam)
            
            if newparam.logP-logp_old + self.proposals.log_J > log(random()):
                newparam.logL = self.user.log_likelihood(newparam)
                
                if newparam.logL > logLmin:
                    oldparam = newparam
                    logp_old = newparam.logP
                    accepted+=1
            
            self.jumps+=1
            
            if self.jumps > self.maxmcmc: break

        self.acceptance = float(accepted)/float(self.jumps)
        self.estimate_nmcmc()
        
        return oldparam
    
    def autotune(self, target = 0.654):
        if self.acceptance < target: self.steps -= 1
        if self.acceptance > target: self.steps += 1
        if self.steps < 1: self.steps = 1
