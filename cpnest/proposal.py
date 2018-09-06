from __future__ import division
from functools import reduce
import numpy as np
from math import log,sqrt,fabs,exp
from abc import ABCMeta,abstractmethod
import random
from random import sample,gauss,randrange,uniform

class Proposal(object):
    """
    Base class for jump proposals
    """
    __metaclass__ = ABCMeta
    log_J = 0.0 # Jacobian of this jump proposal
    @abstractmethod
    def get_sample(self,old):
        """
        Returns a new proposed sample given the old one.
        Must be implemented by user
        """
        pass

class EnsembleProposal(Proposal):
    """
    Base class for ensemble proposals
    """
    ensemble=None
    def set_ensemble(self,ensemble):
        """
        Set the ensemble of points to use
        """
        self.ensemble=ensemble

class ProposalCycle(EnsembleProposal):
    """
    A proposal that cycles through a list of
    jumps.

    Initialisation arguments:
    
    proposals : A list of jump proposals
    weights   : Weights for each type of jump
    
    Optional arguments:
    cyclelength : length of the propsal cycle

    """
    idx=0 # index in the cycle
    N=0   # numer of proposals in the cycle
    def __init__(self,proposals,weights,cyclelength=100,*args,**kwargs):
        super(ProposalCycle,self).__init__(*args,**kwargs)
        assert(len(weights)==len(proposals))
        self.cyclelength = cyclelength
        self.weights = weights
        self.proposals = proposals
        self.set_cycle()

    def set_cycle(self):
        # The cycle is a list of indices for self.proposals
        self.cycle = np.random.choice(self.proposals, size=self.cyclelength,
                                      p=self.weights, replace=True)
        self.N=len(self.cycle)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = self.normalise_weights(weights)

    def normalise_weights(self, weights):
        norm = sum(weights)
        for i, _ in enumerate(weights):
            weights[i]=weights[i] / norm
        return weights

    def get_sample(self,old):
        # Call the current proposal and increment the index
        p = self.cycle[self.idx]
        new = p.get_sample(old)
        self.log_J = p.log_J
        self.idx = (self.idx + 1) % self.N
        return new

    def set_ensemble(self,ensemble):
        # Calls set_ensemble on each proposal that is of ensemble type
        self.ensemble=ensemble
        for p in self.proposals:
            if isinstance(p,EnsembleProposal):
                p.set_ensemble(self.ensemble)

    def add_proposal(self, proposal, weight):
        self.proposals = self.proposals + [proposal]
        self.weights = self.weights + [weight]
        self.set_cycle()


class EnsembleWalk(EnsembleProposal):
    """
    The Ensemble "walk" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65

    Draws a step with the sample covariance of the points
    in the ensemble.
    """
    log_J = 0.0 # Symmetric proposal
    Npoints = 3
    def get_sample(self,old):
        subset = sample(self.ensemble,self.Npoints)
        center_of_mass = reduce(type(old).__add__,subset)/float(self.Npoints)
        out = old
        for x in subset:
            out += (x - center_of_mass)*gauss(0,1)
        return out

class EnsembleStretch(EnsembleProposal):
    """
    The Ensemble "stretch" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65
    """
    def get_sample(self,old):
        scale = 2.0 # Will stretch factor in (1/scale,scale)
        # Pick a random point to move toward
        a = random.choice(self.ensemble)
        # Pick the scale factor
        x = uniform(-1,1)*log(scale)
        Z = exp(x)
        out = a + (old - a)*Z
        # Jacobian
        self.log_J = out.dimension * x
        return out

class DifferentialEvolution(EnsembleProposal):
    """
    Differential evolution move:
    Draws a step by taking the difference vector between two points in the
    ensemble and adding it to the current point.
    See e.g. Exercise 30.12, p.398 in MacKay's book
    http://www.inference.phy.cam.ac.uk/mackay/itila/

    We add a small perturbation around the exact step
    """
    log_J = 0.0 # Symmetric jump
    def get_sample(self,old):
        a,b = sample(self.ensemble,2)
        sigma = 1e-4 # scatter around difference vector by this factor
        out = old + (b-a)*gauss(1.0,sigma)
        return out

class EnsembleEigenVector(EnsembleProposal):
    """
    A jump along a randomly-chosen eigenvector
    of the covariance matrix of the ensemble
    """
    log_J = 0.0
    eigen_values=None
    eigen_vectors=None
    def set_ensemble(self,ensemble):
        """
        Over-ride default set_ensemble so that the
        eigenvectors are recomputed when it is updated
        """
        super(EnsembleEigenVector,self).set_ensemble(ensemble)
        self.update_eigenvectors()

    def update_eigenvectors(self):
        """
        Recompute the eigenvectors of the covariance matrix
        from the ensemble
        """
        n=len(self.ensemble)
        dim = self.ensemble[0].dimension
        cov_array = np.zeros((dim,n))
        if dim == 1:
            name=self.ensemble[0].names[0]
            self.eigen_values = np.atleast_1d(np.var([self.ensemble[j][name] for j in range(n)]))
            self.eigen_vectors = np.eye(1)
        else:	 
            for i,name in enumerate(self.ensemble[0].names):
                for j in range(n): cov_array[i,j] = self.ensemble[j][name]
            covariance = np.cov(cov_array)
            self.eigen_values,self.eigen_vectors = np.linalg.eigh(covariance)

    def get_sample(self,old):
        """
        Propose a jump along a random eigenvector
        """
        out = old
        # pick a random eigenvector
        i = randrange(old.dimension)
        jumpsize = sqrt(fabs(self.eigen_values[i]))*gauss(0,1)
        for k,n in enumerate(out.names):
            out[n]+=jumpsize*self.eigen_vectors[k,i]
        return out


class DefaultProposalCycle(ProposalCycle):
    """
    A default proposal cycle that uses the Walk, Stretch, Differential Evolution
    and Eigenvector proposals
    """
    def __init__(self,*args,**kwargs):
        proposals = [EnsembleWalk(), EnsembleStretch(), DifferentialEvolution(), EnsembleEigenVector()]
        weights = [1.0,1.0,3.0,10.0]
        super(DefaultProposalCycle,self).__init__(proposals,weights,*args,**kwargs)

class HamiltonianProposal(EnsembleProposal):
    """
    Base class for hamiltonian proposals
    """
    L   = 20
    dt	= 1e-2
    mass_matrix = None
    inverse_mass_matrix = None
    momenta_distribution = None
    
    def __init__(self, *args, **kwargs):
        """
        Sets the boundary conditions as a free particle at infinity (V=0)
        """
        super(HamiltonianProposal, self).__init__(*args, **kwargs)
        self.T  = self.kinetic_energy
        self.V  = lambda x: np.zeros(x.shape[0])
        self.dV = lambda x: np.zeros(x.shape[0])
        try:
            self.V = kwargs['potential']
            self.dV = kwargs['force']
            self.estimate_potential = False
        except:
            print("Using empirical potential estimator\n")
            self.estimate_potential = True
    
    def set_ensemble(self, ensemble):
        """
        override the set ensemble method
        to update masses, momenta distribution
        and potential
        """
        super(HamiltonianProposal,self).set_ensemble(ensemble)
        self.update_mass()
        self.update_momenta_distribution()

    def update_potential_energy(self, tracers_array):
        """
        update the potential energy function
        """
        self.V  = dg.Potential(tracers_array.shape[0], tracers_array.T)
        self.dV = self.V.force

    def update_momenta_distribution(self):
        """
        update the momenta distribution
        """
        self.momenta_distribution = multivariate_normal(cov=self.mass_matrix)

    def update_mass(self):
        """
        Recompute the mass matrix (covariance matrix)
        from the ensemble
        """
        n   = len(self.ensemble)
        dim = self.ensemble[0].dimension
        cov_array = np.zeros((dim,n))
        
        if dim == 1:
            name=self.ensemble[0].names[0]
            cov_array = np.atleast_2d([self.ensemble[j][name] for j in range(n)])
            self.mass_matrix = np.atleast_1d(np.var([self.ensemble[j][name] for j in range(n)]))
            self.inverse_mass_matrix = 1./self.mass_matrix
        else:
            for i,name in enumerate(self.ensemble[0].names):
                for j in range(n): cov_array[i,j] = self.ensemble[j][name]
                covariance = np.cov(cov_array)
                self.mass_matrix = np.linalg.inv(covariance)
                self.inverse_mass_matrix = covariance

        # update the potential energy estimate
        if self.estimate_potential: self.update_potential_energy(cov_array)

    def kinetic_energy(self,p):
        """
        kinetic energy part for the Hamiltonian
        """
        return 0.5 * np.dot(p,np.dot(self.inverse_mass_matrix,p))

class LeapFrog(HamiltonianProposal):
    """
    Leap frog integrator proposal for an uncostrained
    Hamiltonian Monte Carlo step
    """
    # symmetric proposal
    log_J = 0.0
    def get_sample(self, old):
        # transform into a numpy array for flexibility
        old_arr = old.asnparray()
        q0 = old_arr.view(dtype=np.float64)[:-2]
        # generate a canonical momentum
        p0 = np.atleast_1d(self.momenta_distribution.rvs())
        # evolve along the trajectory
        q, p = self.evolve_trajectory(p0, q0)
        
        initial_energy = self.T(p0) + self.V(q0)
        final_energy   = self.T(p)  + self.V(q)
        
        dE = min(0.0, initial_energy - final_energy)
        if dE > log(np.random.uniform()):
            # accept
            for j,k in enumerate(old.names):
                old[k] = q[j]
        return old
    
    def evolve_trajectory(self, p0, q0):
        """
        https://arxiv.org/pdf/1005.0157.pdf
        https://arxiv.org/pdf/1206.1901.pdf
        """
        
        self.dt = np.abs(np.random.normal(1e-2,1e-3))
        self.L  = np.random.randint(20,50)
        #f = open('trajectory.txt','a')
        # Updating the momentum a half-step
        p = p0-0.5 * self.dt * self.dV(q0)
        q = q0
        
        invM = np.squeeze(np.diag(self.inverse_mass_matrix))
        
        #f.write("%e %e %e %e\n"%(q0,p,self.V(q0),self.dV(q0)))
        for i in xrange(self.L):
            
            # do a step
            q += self.dt * p * invM
            dV = self.dV(q)
            # take a full momentum step
            p += - self.dt * dV
        #	f.write("%e %e %e %e\n"%(q,p,self.V(q),dV))
        # Do a final update of the momentum for a half step
        p += - 0.5 * self.dt * dV
        #f.write("%e %e %e %e\n"%(q,p,self.V(q),dV))
        #f.close()
        #exit()
        return q, -p
