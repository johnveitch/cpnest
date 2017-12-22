from __future__ import division
import sys
import os
import numpy as np
from math import log
from collections import deque
from random import random,randrange
from operator import attrgetter
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, LSQUnivariateSpline
from scipy.stats import multivariate_normal

from . import parameter
from . import proposal
from sampler import Sampler

class HMCSampler(Sampler):
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
    def __init__(self, *args, l=1.0, **kwargs):
        super(HMCSampler,self).__init__(*args, **kwargs)
        self.step = self.constrained_leapfrog_step
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
        super(HMCSampler, self).reset()

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
    
    def autotune(self, target = 0.654):
        if self.acceptance < target: self.steps -= 1
        if self.acceptance > target: self.steps += 1
        if self.steps < 1: self.steps = 1

