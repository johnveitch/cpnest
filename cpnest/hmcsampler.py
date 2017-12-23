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
from . import dpgmm

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
    def __init__(self, *args, **kwargs):
        self.l              = kwargs.pop('l')
        super(HMCSampler,self).__init__(*args, **kwargs)
        self.step = self.constrained_leapfrog_step
        # step size choice from http://www.homepages.ucl.ac.uk/~ucakabe/papers/Bernoulli_11b.pdf
        # which asks for the relation step_size = l * dims**(1/4)
        self.step_size      = 1.0
        self.steps          = 20
        self.momenta_distribution = None
        self.counter = 0
    
    def reset(self):
        """
        Initialise the sampler
        """
        super(HMCSampler, self).reset()
        
        self.step_size = self.l * float(len(self.evolution_points[0].names))**(0.25)

        # estimate the initial gradients
        if self.verbose > 2: sys.stderr.write("Computing initial gradients ...")
        self.potential  = self.estimate_gradient()
        self.constraint = self.potential
        
        if self.verbose > 2: sys.stderr.write("done\n")
        self.initialised=True
    
    def estimate_gradient(self):
        # arrange the samples in a poolsize x dim data matrix
        x = np.array([[self.evolution_points[i][key] for i in range(len(self.evolution_points))] for j,key in enumerate(self.evolution_points[0].names)])
        dim = len(self.evolution_points[0].names)
        self.mass_matrix = np.atleast_1d(np.cov(x))
        if self.mass_matrix.shape[0] > 1:
            self.inverse_mass_matrix = np.linalg.inv(self.mass_matrix)
        else:
            self.inverse_mass_matrix = np.atleast_2d(1./self.mass_matrix)
        # estimate the potential
        self.potential = dpgmm.gradient.Potential(dim, x.T)
        self.momenta_distribution = multivariate_normal(cov=self.mass_matrix)
        return self.potential

    def gradient(self, inParam):
        x = inParam.asnparray()
        x = x.view(dtype=np.float64)[:-2]
        return self.potential.force(x)

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

        return 0.5 * np.dot(p,np.dot(self.inverse_mass_matrix,p))

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

    def constrained_leapfrog_step(self, position):
        """
        https://arxiv.org/pdf/1005.0157.pdf
        https://arxiv.org/pdf/1206.1901.pdf
        """

        momentum = self.user.new_point()
        v = np.atleast_1d(self.momenta_distribution.rvs())
        g = self.gradient(position)

        # Updating the momentum a half-step
        for j,k in enumerate(self.evolution_points[0].names):
            momentum[k] = v[j]-0.5 * self.step_size * g[j]

        for i in xrange(self.steps):
            
            # do a step
            for j,k in enumerate(self.evolution_points[0].names):
                position[k] += self.step_size * momentum[k] * self.inverse_mass_matrix[j,j]
        
            # if the trajectory brings us outside the prior boundary, bounce back and forth
            # see https://arxiv.org/pdf/1206.1901.pdf pag. 37
            
            position.logP = self.user.log_prior(position)
            
            if not(np.isfinite(position.logP)):
                return position, momentum
                for j,k in enumerate(self.evolution_points[0].names):
                    while position[k] > self.user.bounds[j][1]:
                        position[k] = self.user.bounds[j][1] - (position[k] - self.user.bounds[j][1])
                        momentum[k] = -momentum[k]
                    while position[k] < self.user.bounds[j][0]:
                        position[k] = self.user.bounds[j][0] + (self.user.bounds[j][0] - position[k])
                        momentum[k] = -momentum[k]

            # Update gradient
            g = self.gradient(position)
#            take a full momentum step
            for j,k in enumerate(self.evolution_points[0].names):
                momentum[k] += - self.step_size * g[j]
            # compute the constraint
            position.logL = self.user.log_likelihood(position)

#            # check on the constraint
#            if position.logL > logLmin:
#                # take a full momentum step
#                for j,k in enumerate(self.evolution_points[0].names):
#                    momentum[k] += - self.step_size * g[j]
#            else:
#                # compute the normal to the constraint
#                gL = self.constrain(position)
#                n = gL/np.sqrt(np.sum(gL**2))
#
#                # bounce on the constraint
#                for j,k in enumerate(self.evolution_points[0].names):
#                    momentum[k] += - 2 * (momentum[k]*n[j]) * n[j]

        # Do a final update of the momentum for a half step
        g = self.gradient(position)
        for j,k in enumerate(self.evolution_points[0].names):
            momentum[k] += - 0.5 * self.step_size * g[j]
        return position
#    , momentum
