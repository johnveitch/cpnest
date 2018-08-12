from __future__ import division
from functools import reduce
import numpy as np
from math import log,sqrt,fabs,exp
from abc import ABCMeta,abstractmethod
import random
from random import sample,gauss,randrange,uniform
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal

class Proposal(object):
    """
    Base class for jump proposals
    """
    __metaclass__ = ABCMeta
    log_J = 0.0 # Jacobian of this jump proposal
    @abstractmethod
    def get_sample(self,old,**kwargs):
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
    cyclelength : length of the proposal cycle

    """
    idx=0 # index in the cycle
    N=0   # number of proposals in the cycle
    def __init__(self,proposals,weights,cyclelength=100,*args,**kwargs):
        super(ProposalCycle,self).__init__()
        assert(len(weights)==len(proposals))
        # Normalise the weights
        norm = sum(weights)
        for i,_ in enumerate(weights):
            weights[i]=weights[i]/norm
        self.proposals=proposals
        # The cycle is a list of indices for self.proposals
        self.cycle = np.random.choice(self.proposals, size = cyclelength, p=weights, replace=True)
        self.N=len(self.cycle)

    def get_sample(self,old,**kwargs):
        # Call the current proposal and increment the index
        p = self.cycle[self.idx]
        new = p.get_sample(old,**kwargs)
        self.log_J = p.log_J
        self.idx = (self.idx + 1) % self.N
        return new

    def set_ensemble(self,ensemble):
        # Calls set_ensemble on each proposal that is of ensemble type
        self.ensemble=ensemble
        for p in self.proposals:
            if isinstance(p,EnsembleProposal):
                p.set_ensemble(self.ensemble)

class EnsembleWalk(EnsembleProposal):
    """
    The Ensemble "walk" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65

    Draws a step with the sample covariance of the points
    in the ensemble.
    """
    log_J = 0.0 # Symmetric proposal
    Npoints = 3
    def get_sample(self,old,**kwargs):
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
    def get_sample(self,old,**kwargs):
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
    def get_sample(self,old,**kwargs):
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

    def get_sample(self,old,**kwargs):
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
    and Eigenvector proposals. If the user passes a force function and/or a
    potential barrier, then a leap frog hamiltonian proposal is also added
    """
    def __init__(self,*args,**kwargs):
        
        proposals = [EnsembleWalk(),
                     EnsembleStretch(),
                     DifferentialEvolution(),
                     EnsembleEigenVector()]
        weights = [0.2,
                   0.2,
                   0.2,
                   0.1]
        if kwargs is not None:
            # check if the user has defined a force function and a potential barrier
            if 'force' in kwargs and 'barrier' in kwargs:
                proposals.append(ConstrainedLeapFrog(**kwargs))#
                weights.append(0.2)
                proposals.append(LeapFrog(**kwargs))
                weights.append(0.1)
            elif 'force' in kwargs:
                proposals.append(LeapFrog(**kwargs))
                weights.append(0.1)
#        proposals = [ConstrainedLeapFrog(**kwargs),LeapFrog(**kwargs)]
#        weights = [1,1]
        super(DefaultProposalCycle,self).__init__(proposals,weights,*args,**kwargs)

class HamiltonianProposal(EnsembleProposal):
    """
    Base class for hamiltonian proposals
    """
    mass_matrix = None
    inverse_mass_matrix = None
    momenta_distribution = None
    
    def __init__(self, *args, **kwargs):
        """
        Sets the boundary conditions as a free particle at infinity (V=0)
        """
        super(HamiltonianProposal, self).__init__()
        self.T  = self.kinetic_energy
        self.dV = kwargs['force']
        self.V = kwargs['potential']
        self.C = kwargs['barrier']
        self.normal = None
    
    def set_ensemble(self, ensemble):
        """
        override the set ensemble method
        to update masses and momenta distribution
        """
        super(HamiltonianProposal,self).set_ensemble(ensemble)
        self.update_mass()
        self.update_momenta_distribution()

    def update_normal_vector(self, tracers_array, V_vals):
        """
        update the constraint by approximating the
        loglikelihood hypersurface as a spline in
        each dimension.
        This is a rough approximation which
        improves as the algorithm proceeds
        """
        self.normal = []
        for i,x in enumerate(tracers_array):
            # sort the values
            idx = x.argsort()
            xs = x[idx]
            Vs = V_vals[idx]
            # remove potential duplicate entries
            xs, ids = np.unique(xs, return_index = True)
            Vs = Vs[ids]
            # pick only finite values
            idx = np.isfinite(Vs)
            # pick fixed n-dim knots for the spline from the percentiles
            knots = np.percentile(xs[idx],np.linspace(1,99,tracers_array.shape[0]))
            # construct a LSQ spline interpolant, weighting potential nans as zeros (which should have been sorted out already
            window_length = len(idx)//2+1
            if window_length%2 == 0: window_length += 1
            f = savgol_filter(Vs[idx], window_length, 5  , deriv=1, delta=0.01, mode='mirror')
            self.normal.append(LSQUnivariateSpline(xs[idx], f, knots, ext = 3, k = 3))
#            np.savetxt('dlogL_spline_%d.txt'%i,np.column_stack((xs[idx],self.normal[-1](xs[idx]),f)))

    def unit_normal(self, x):
        """
        Returns the unit normal to the constraint surface
        obtained from the spline interpolation of the
        directional derivatives of the likelihood
        """
        v = np.array([self.normal[i](x[n]) for i,n in enumerate(x.names)])
        v[np.isnan(v)] = -1.0
        n = v/np.linalg.norm(v)
        return n

    def gradient(self, x):
        """
        return the gradient of the potential function as numpy ndarray
        """
        dV = self.dV(x)
        return dV.view(np.float64)
    
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
            self.inverse_mass_matrix = np.atleast_1d(np.var([self.ensemble[j][name] for j in range(n)]))
            pvals = np.atleast_1d([self.ensemble[j].logL for j in range(n)])
            self.mass_matrix = 1./self.inverse_mass_matrix
        else:
            for i,name in enumerate(self.ensemble[0].names):
                for j in range(n):
                    cov_array[i,j] = self.ensemble[j][name]
            pvals = np.array([self.ensemble[j].logL for j in range(n)])
            covariance = np.cov(cov_array)
            self.mass_matrix = np.diag(np.diag(np.linalg.inv(covariance)))
            self.inverse_mass_matrix = np.diag(np.diag(covariance))

        # update the potential energy estimate
        self.update_normal_vector(cov_array, pvals)

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
    def get_sample(self, q0, **kwargs):
        # generate a canonical momentum
        p0 = np.atleast_1d(self.momenta_distribution.rvs())
        # evolve along the trajectory
        q, p = self.evolve_trajectory(p0, q0, constraint=kwargs['constraint'])
        
        initial_energy = self.T(p0) + self.V(q0)
        final_energy   = self.T(p)  + self.V(q)
        log_J = min(0.0, initial_energy - final_energy)
        return q
    
    def evolve_trajectory(self, p0, q0, constraint=None):
        """
        https://arxiv.org/pdf/1206.1901.pdf
        """
        invM = np.atleast_1d(np.squeeze(np.diag(self.inverse_mass_matrix)))
        # generate the trajectory lengths from a scale invariant distribution
        self.L  = int(np.exp(np.random.uniform(np.log(10),np.log(50))))
        self.dt = 3e-2*float(len(invM))**(-0.25)
        self.dt = np.abs(gauss(self.dt,self.dt))
        # Updating the momentum a half-step
        p = p0-0.5 * self.dt * self.gradient(q0)
        q = q0
        for i in range(self.L):
            # do a step
            for j,k in enumerate(q.names):
                u,l = constraint[j][1], constraint[j][0]
                q[k] += self.dt * p[j] * invM[j]
                # check and reflect against the bounds
                # of the allowed parameter range
                if q[k] > u:
                    q[k] = u - (q[k] - u)
                    p[j] *= -1
                if q[k] < l:
                    q[k] = l + (l - q[k])
                    p[j] *= -1
        
            dV = self.gradient(q)
            # take a full momentum step
            p += - self.dt * dV
        # Do a final update of the momentum for a half step
        p += - 0.5 * self.dt * dV

        return q, -p

class ConstrainedLeapFrog(HamiltonianProposal):
    """
    Leap frog integrator proposal for a costrained
    (logLmin defines a reflective boundary)
    Hamiltonian Monte Carlo step
    """
    # symmetric proposal
    log_J = 0.0
    def get_sample(self, q0, **kwargs):
        # generate a canonical momentum
        p0 = np.atleast_1d(self.momenta_distribution.rvs())
        # evolve along the trajectory
        q, p = self.evolve_trajectory(p0, q0, barrier=kwargs['barrier'], constraint=kwargs['constraint'])
        
        initial_energy = self.T(p0) + self.V(q0)
        final_energy   = self.T(p)  + self.V(q)
        log_J = min(0.0, initial_energy - final_energy)
        return q
    
    def evolve_trajectory(self, p0, q0, barrier=-np.inf, constraint=None):
        """
        https://arxiv.org/pdf/1005.0157.pdf
        """
        invM = np.atleast_1d(np.squeeze(np.diag(self.inverse_mass_matrix)))
#        f = open("mass.txt","a")
#        for M in invM: f.write("%e\t"%(1./M))
#        f.write("\n")
#        f.close()
        # generate the trajectory lengths from a scale invariant distribution
        self.L  = int(np.exp(np.random.uniform(np.log(10),np.log(50))))
        self.dt = 3e-2*float(len(invM))**(-0.25)
        self.dt = np.abs(gauss(self.dt,self.dt))
#        f = open("trajectory.txt","w")
#        for j,k in enumerate(q0.names):
#            f.write("%s\t"%k)
#        f.write("barrier\t")
#        f.write("logL\n")
#        # Updating the momentum a half-step
#        for j,k in enumerate(q0.names):
#            f.write("%e\t"%q0[k])
#        f.write("%e\t"%barrier)
#        f.write("%e\n"%q0.logL)

        p = p0-0.5 * self.dt * self.gradient(q0)
        q = q0
#        for j,k in enumerate(q.names):
#            f.write("%e\t"%q[k])
#        f.write("%e\t"%barrier)
#        f.write("%e\n"%q.logL)
        for i in range(self.L):
            # do a step
            for j,k in enumerate(q.names):
                u,l = constraint[j][1], constraint[j][0]
                q[k] += self.dt * p[j] * invM[j]
                # check and reflect against the bounds
                # of the allowed parameter range
                if q[k] > u:
                    q[k] = u - (q[k] - u)
                    p[j] *= -1
                if q[k] < l:
                    q[k] = l + (l - q[k])
                    p[j] *= -1
            dV = self.gradient(q)
            
            # if the trajectory led us to a lower likelihood,
            # reflect the momentum orthogonally to the surface
            if np.isfinite(barrier):
                logL = self.C(q)
                q.logL = logL
                
                if (logL - barrier) <= 0:
                    normal = self.unit_normal(q)
                    p = p - 2.0*np.dot(p,normal)*normal
                else:
                    # take a full momentum step
                    p += - self.dt * dV

#            for j,k in enumerate(q.names):
#                f.write("%e\t"%q[k])
#            f.write("%e\t"%barrier)
#            f.write("%e\n"%q.logL)
        # Do a final update of the momentum for a half step
        p += - 0.5 * self.dt * dV
#        f.close()
        return q, -p
