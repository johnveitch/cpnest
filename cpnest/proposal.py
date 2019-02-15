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
    Base abstract class for jump proposals
    """
    __metaclass__ = ABCMeta
    log_J = 0.0 # Jacobian of this jump proposal
    @abstractmethod
    def get_sample(self,old):
        """
        Returns a new proposed sample given the old one.
        Must be implemented by user

        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`
        
        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
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
    cyclelength : length of the proposal cycle. Default: 100

    """
    idx=0 # index in the cycle
    N=0   # number of proposals in the cycle
    def __init__(self,proposals,weights,cyclelength=100,*args,**kwargs):
        super(ProposalCycle,self).__init__()
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

    def get_sample(self,old,**kwargs):
        # Call the current proposal and increment the index
        self.idx = (self.idx + 1) % self.N
        p = self.cycle[self.idx]
        new = p.get_sample(old,**kwargs)
        self.log_J = p.log_J
        return new

    def set_ensemble(self,ensemble):
        """
        Updates the ensemble statistics
        by calling it on each :obj:`EnsembleProposal`
        """
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

    Draws a step by evolving along the
    direction of the center of mass of
    3 points in the ensemble.
    """
    log_J = 0.0 # Symmetric proposal
    Npoints = 3
    def get_sample(self,old):
        """
        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`
        
        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
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
        """
        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`
        
        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
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
        """
        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`
        
        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
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
    covariance=None
    def set_ensemble(self,ensemble):
        """
        Over-ride default set_ensemble so that the
        eigenvectors are recomputed when it is updated
        """
        super(EnsembleEigenVector,self).set_ensemble(ensemble)
        self.update_eigenvectors()

    def update_eigenvectors(self):
        """
        Recompute the eigenvectors and eigevalues
        of the covariance matrix of the ensemble
        """
        n=len(self.ensemble)
        dim = self.ensemble[0].dimension
        cov_array = np.zeros((dim,n))
        if dim == 1:
            name=self.ensemble[0].names[0]
            self.eigen_values = np.atleast_1d(np.var([self.ensemble[j][name] for j in range(n)]))
            self.covariance = self.eigen_values
            self.eigen_vectors = np.eye(1)
        else:	 
            for i,name in enumerate(self.ensemble[0].names):
                for j in range(n): cov_array[i,j] = self.ensemble[j][name]
            self.covariance = np.cov(cov_array)
            self.eigen_values,self.eigen_vectors = np.linalg.eigh(self.covariance)

    def get_sample(self,old):
        """
        Propose a jump along a random eigenvector
        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`
        
        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
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
    A default proposal cycle that uses the
    :obj:`cpnest.proposal.EnsembleWalk`, :obj:`cpnest.proposal.EnsembleStretch`,
    :obj:`cpnest.proposal.DifferentialEvolution`, :obj:`cpnest.proposal.EnsembleEigenVector`
    ensemble proposals.
    """
    def __init__(self):
        
        proposals = [EnsembleWalk(),
                     EnsembleStretch(),
                     DifferentialEvolution(),
                     EnsembleEigenVector()]
        weights = [2,
                   2,
                   5,
                   1]
        super(DefaultProposalCycle,self).__init__(proposals, weights)

class HamiltonianProposalCycle(ProposalCycle):
    def __init__(self, model=None):
        """
        A proposal cycle that uses the hamiltonian :obj:`ConstrainedLeapFrog`
        proposal.
        Requires a :obj:`cpnest.Model` to be passed for access to the user-defined
        :obj:`cpnest.Model.force` (the gradient of :obj:`cpnest.Model.potential`) and
        :obj:`cpnest.Model.log_likelihood` to define the reflective
        """
        weights = [1]
        proposals = [ConstrainedLeapFrog(model=model)]
        super(HamiltonianProposalCycle,self).__init__(proposals, weights)

class HamiltonianProposal(EnsembleEigenVector):
    """
    Base class for hamiltonian proposals
    """
    mass_matrix = None
    inverse_mass_matrix = None
    momenta_distribution = None
    
    def __init__(self, model=None, **kwargs):
        """
        Initialises the class with the kinetic
        energy and the :obj:`cpnest.Model.potential`.
        """
        super(HamiltonianProposal, self).__init__(**kwargs)
        self.T      = self.kinetic_energy
        self.V      = model.potential
        self.normal = None
        self.dt     = 5e-1
        self.dtmin  = 1e-3
        self.dtmax  = 1.0
        self.TARGET = 0.654
    
    def set_ensemble(self, ensemble):
        """
        override the set ensemble method
        to update masses, momenta distribution
        and to heuristically estimate the normal vector to the
        hard boundary defined by logLmin.
        """
        super(HamiltonianProposal,self).set_ensemble(ensemble)
        self.update_mass()
        self.update_normal_vector()
        self.update_momenta_distribution()
    
    def update_normal_vector(self):
        """
        update the constraint by approximating the
        loglikelihood hypersurface as a spline in
        each dimension.
        This is an approximation which
        improves as the algorithm proceeds
        """
        n = self.ensemble[0].dimension
        tracers_array = np.zeros((len(self.ensemble),n))
        for i,samp in enumerate(self.ensemble):
            tracers_array[i,:] = samp.values
        V_vals = np.atleast_1d([p.logL for p in self.ensemble])
        
        self.normal = []
        for i,x in enumerate(tracers_array.T):
            # sort the values
            idx = x.argsort()
            xs = x[idx]
            Vs = V_vals[idx]
            # remove potential duplicate entries
            xs, ids = np.unique(xs, return_index = True)
            Vs = Vs[ids]
            # pick only finite values
            idx = np.isfinite(Vs)
            Vs  = Vs[idx]
            xs  = xs[idx]
            # filter to within the 90% range of the Pvals
            Vl,Vh = np.percentile(Vs,[5,95])
            (idx,) = np.where(np.logical_and(Vs > Vl,Vs < Vh))
            Vs = Vs[idx]
            xs = xs[idx]
            # Pick knots for this parameters: Choose 5 knots between
            # the 1st and 99th percentiles (heuristic tuning WDP)
            knots = np.percentile(xs,np.linspace(1,99,5))
            # Guesstimate the length scale for numerical derivatives
            dimwidth = knots[-1]-knots[0]
            delta = 0.1 * dimwidth / len(idx)
            # Apply a Savtzky-Golay filter to the likelihoods (low-pass filter)
            window_length = len(idx)//2+1 # Window for Savtzky-Golay filter
            if window_length%2 == 0: window_length += 1
            f = savgol_filter(Vs, window_length,
                              5, # Order of polynominal filter
                              deriv=1, # Take first derivative
                              delta=delta, # delta for numerical deriv
                              mode='mirror' # Reflective boundary conds.
                              )
            # construct a LSQ spline interpolant
            self.normal.append(LSQUnivariateSpline(xs, f, knots, ext = 3, k = 3))
#            np.savetxt('dlogL_spline_%d.txt'%i,np.column_stack((xs,Vs,self.normal[-1](xs),f)))

    def unit_normal(self, q):
        """
        Returns the unit normal to the iso-Likelihood surface
        at x, obtained from the spline interpolation of the
        directional derivatives of the likelihood
        Parameters
        ----------
        q : :obj:`cpnest.parameter.LivePoint`
            position
        
        Returns
        ----------
        n: :obj:`numpy.ndarray` unit normal to the logLmin contour evaluated at q
        """
        v               = np.array([self.normal[i](q[n]) for i,n in enumerate(q.names)])
        v[np.isnan(v)]  = -1.0
        n               = v/np.linalg.norm(v)
        return n

    def gradient(self, q):
        """
        return the gradient of the potential function as numpy ndarray
        Parameters
        ----------
        q : :obj:`cpnest.parameter.LivePoint`
            position
        
        Returns
        ----------
        dV: :obj:`numpy.ndarray` gradient evaluated at q
        """
        dV = self.dV(q)
        return dV.view(np.float64)
    
    def update_momenta_distribution(self):
        """
        update the momenta distribution using the
        mass matrix (precision matrix of the ensemble).
        """
        self.momenta_distribution = multivariate_normal(cov=self.mass_matrix)#p.identity(self.d))#

    def update_mass(self):
        """
        Update the mass matrix (covariance matrix) and
        inverse mass matrix (precision matrix)
        from the ensemble, allowing for correlated momenta
        """
        self.inverse_mass_matrix    = np.atleast_2d(np.diag(np.diag(self.covariance)))
        self.mass_matrix            = np.linalg.inv(self.inverse_mass_matrix)
        self.inverse_mass           = np.atleast_1d(np.squeeze(np.diag(self.inverse_mass_matrix)))
        self.d                      = len(self.inverse_mass)
        self.set_trajectory_length()
    
    def set_time_step(self):
        """
        Set the initial time step
        Step size: dimension^-(1/4) based on
        http://www.homepages.ucl.ac.uk/~ucakabe/papers/Bernoulli_11b.pdf
        """
        self.dt = 5e-3
    
    def update_time_step(self, acceptance):
        """
        Update the time step according to the
        acceptance rate
        Parameters
        ----------
        acceptance : :obj:'numpy.float'
        """
        if acceptance <= self.TARGET:
            self.dt *= 0.95
        else:
            self.dt *= 1.05

        if self.dt > self.dtmax: self.dt = self.dtmax
        if self.dt < self.dtmin: self.dt = self.dtmin
#        print('dt:',self.dt,'L:',self.L)
    def set_trajectory_length(self):
        """
        Set the trajectory length based on the mean parameters
        range and dimensionality so that, depending of the current time step,
        we can in principle traverse the whole current allowed region
        """
        self.L = np.random.randint(10,50)#int(np.ceil(self.d**(0.25))+(np.mean(self.inverse_mass)/self.dt)**(0.25))

    def kinetic_energy(self,p):
        """
        kinetic energy part for the Hamiltonian.
        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            momentum
        
        Returns
        ----------
        T: :float: kinetic energy
        """
        return 0.5 * np.dot(p,np.dot(self.inverse_mass_matrix,p))

class LeapFrog(HamiltonianProposal):
    """
    Leap frog integrator proposal for an unconstrained
    Hamiltonian Monte Carlo step
    """
    def __init__(self, model=None, **kwargs):
        """
        Parameters
        ----------
        model : :obj:`cpnest.Model`
        """
        super(LeapFrog, self).__init__(model=model, **kwargs)
        self.dV             = model.force
        self.prior_bounds   = model.bounds

    def get_sample(self, q0, *args):
        """
        Propose a new sample, starting at q0
        
        Parameters
        ----------
        q0 : :obj:`cpnest.parameter.LivePoint`
            position
        
        Returns
        ----------
        q: :obj:`cpnest.parameter.LivePoint`
            position
        """
        # generate a canonical momentum
        p0 = np.atleast_1d(self.momenta_distribution.rvs())
        T0 = self.T(p0)
        V0 = -q0.logP
        # evolve along the trajectory
        q, p = self.evolve_trajectory(p0, q0, *args)
        # minus sign from the definition of the potential
        initial_energy = T0 + V0
        final_energy   = self.T(p)  - q.logP
        self.log_J = min(0.0, initial_energy - final_energy)
        return q
    
    def evolve_trajectory(self, p0, q0, *args):
        """
        Hamiltonian leap frog trajectory subject to the
        hard boundary defined by the parameters prior bounds.
        https://arxiv.org/pdf/1206.1901.pdf
        
        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`cpnest.parameter.LivePoint`
            position
        
        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`cpnest.parameter.LivePoint`
            position
        """
        # Updating the momentum a half-step
        p = p0-0.5 * self.dt * self.gradient(q0)
        q = q0.copy()
        
        for i in range(self.L):
            # do a step
            for j,k in enumerate(q.names):
                u,l = self.prior_bounds[j][1], self.prior_bounds[j][0]
                q[k] += self.dt * p[j] * self.inverse_mass[j]
                # check and reflect against the bounds
                # of the allowed parameter range
                while q[k] <= l or q[k] >= u:
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

class ConstrainedLeapFrog(LeapFrog):
    """
    Leap frog integrator proposal for a costrained
    (logLmin defines a reflective boundary)
    Hamiltonian Monte Carlo step.
    """
    def __init__(self, model=None, **kwargs):
        """
        Parameters
        ----------
        model : :obj:`cpnest.Model`
        """
        super(ConstrainedLeapFrog, self).__init__(model=model, **kwargs)
        self.log_likelihood = model.log_likelihood

    def get_sample(self, q0, logLmin=-np.inf):
        """
        Generate new sample with constrained HMC, starting at q0.
        
        Parameters
        ----------
        q0 : :obj:`cpnest.parameter.LivePoint`
            position
        
        logLmin: hard likelihood boundary
        
        Returns
        ----------
        q: :obj:`cpnest.parameter.LivePoint`
            position
        """
        return super(ConstrainedLeapFrog,self).get_sample(q0, logLmin)
    
    def evolve_trajectory(self, p0, q0, logLmin):
        """
        Evolve point according to Hamiltonian method in
        https://arxiv.org/pdf/1005.0157.pdf
        
        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`cpnest.parameter.LivePoint`
            position
        
        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`cpnest.parameter.LivePoint` position
        """
        
        p = p0 - 0.5 * self.dt * self.gradient(q0)
        q = q0.copy()
        i = 0
        last_reflection = 0
#        print("proposal q0.logL, logLmin",q0.logL, logLmin)
        while (i < self.L) or (last_reflection):
            # do a full step in position
            for j,k in enumerate(q.names):
                u, l  = self.prior_bounds[j][1], self.prior_bounds[j][0]
                q[k] += self.dt * p[j] * self.inverse_mass[j]
                # check and reflect against the bounds
                # of the allowed parameter range
                while q[k] <= l or q[k] >= u:
                    if q[k] > u:
                        q[k] = u - (q[k] - u)
                        p[j] *= -1
                    if q[k] < l:
                        q[k] = l + (l - q[k])
                        p[j] *= -1
        
            dV      = self.gradient(q)
            q.logP  = -self.V(q)
            q.logL  = self.log_likelihood(q)
#            print(i,"--> proposal q.logL, logLmin",q.logL, logLmin)
            if np.isfinite(logLmin):
                # if the trajectory led us outside the likelihood bound,
                # reflect the momentum orthogonally to the surface
                if (q.logL - logLmin) <= 0:
                    normal      = self.unit_normal(q)
#                    print(i,"--> reflection p, normal, jump",p, normal, self.dt * p * self.inverse_mass)
                    p           = p - 2.0*np.dot(p,normal)*normal
                    last_reflection = 1
#                    print(i,"--> reflected p, normal, jump",p, normal, self.dt * p * self.inverse_mass)
                else:
                    last_reflection = 0
                    # take a full momentum step
                    p += - self.dt * dV
            else:
                p += - self.dt * dV
            i += 1
            if i == 5*self.L:
                break
        # Do a final update of the momentum for a half step
        p += - 0.5 * self.dt * dV
        return q, -p
