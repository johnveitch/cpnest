from __future__ import division
from functools import reduce
import numpy as np
from math import log,sqrt,fabs,exp
from abc import ABCMeta,abstractmethod
import random
from random import sample,gauss,randrange,uniform
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

#try:
#    from jax import grad, jit
#    FINITEDIFFERENCING = False
#except ImportError:
#    FINITEDIFFERENCING = True
#    pass

FINITEDIFFERENCING = True
from .numerical_grad import numerical_gradient

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
        self.ensemble = ensemble

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
        old: :obj:`cpnest.parameter.LivePoint`
        
        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
        subset = sample(list(self.ensemble),self.Npoints)
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
        old: :obj:`cpnest.parameter.LivePoint`
        
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
    """
    log_J = 0.0 # Symmetric jump
    def get_sample(self,old):
        """
        Parameters
        ----------
        old: :obj:`cpnest.parameter.LivePoint`
        
        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
        a,b = sample(list(self.ensemble),2)
        out = old + (b-a)
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
                   2,
                   10]
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

class HamiltonianProposal(EnsembleProposal):
    """
    Base class for hamiltonian proposals
    """
    def __init__(self, model=None, **kwargs):
        """
        Initialises the class with the kinetic
        energy and the :obj:`cpnest.Model.potential`.
        """
        super(HamiltonianProposal, self).__init__(**kwargs)
        self.model                  = model
        self.T                      = self.kinetic_energy
        self.V                      = model.potential
        self.normal                 = None
        self.scale                  = 1.0
        self.TARGET                 = 0.65
        self.c                      = self.counter()
        self.DEBUG                  = 0
        self.d                      = len(self.model.names)
        self.inverse_mass_matrix    = np.identity(self.d)
        self.mass_matrix            = np.identity(self.d)
        self.inverse_mass           = np.ones(self.d)
        self.logdeterminant         = 0.0
        self.momenta_distribution   = multivariate_normal(cov=self.mass_matrix)#
        self.dt, self.L             = self.set_integration_parameters()
        self.time_step              = DualAveragingStepSize(self.dt,
                                                            target_accept = self.TARGET,
                                                            gamma         = 0.5,
                                                            t0            = 10.0,
                                                            kappa         = 0.75)
        
        self.finite_differencing    = FINITEDIFFERENCING
        if self.finite_differencing == False:
            self.constraint         = grad(self.model.log_likelihood)
        else:
            self.constraint         = numerical_gradient(self.d, self.model.log_likelihood, cache_size = 1000)
            
    def set_ensemble(self,ensemble):
        """
        Over-ride default set_ensemble so that the
        eigenvectors are recomputed when it is updated
        """
        super(HamiltonianProposal,self).set_ensemble(ensemble)
        # compute the metric
        n   = len(self.ensemble)
        cov_array = np.zeros((self.d,n))
        if self.d == 1:
            name=self.ensemble[0].names[0]
            self.inverse_mass_matrix = np.atleast_1d(np.var([self.ensemble[j][name] for j in range(n)]))
            self.mass_matrix         = np.linalg.inv(self.inverse_mass_matrix)
            self.logdeterminant      = self.inverse_mass_matrix[0]
        else:
            for i,name in enumerate(self.ensemble[0].names):
                for j in range(n):
                    cov_array[i,j] = self.ensemble[j][name]
            self.inverse_mass_matrix = np.cov(cov_array)
            self.mass_matrix         = np.linalg.inv(self.inverse_mass_matrix)
            self.logdeterminant      = np.linalg.slogdet(self.mass_matrix)[1]
#        print('Inverse mass:',self.inverse_mass_matrix,'mass:',self.mass_matrix,self.logdeterminant)
        # update the momenta distribution
        self.momenta_distribution    = multivariate_normal(cov=self.mass_matrix)
        
    def counter(self):
        n = 0
        while True:
            yield n
            n += 1
            
    def set_integration_parameters(self):
        """
        Finds the smallest and largest dimensions and use them
        to set the time step and trajectory length
        
        Returns
        ----------
        base_dt: :obj:`float`
        base_L:  :obj:`int`
        """
        
        log_prior_cube_volume = 0.0
        for b in self.model.bounds: log_prior_cube_volume += np.log(b[1]-b[0])
        
        dt   = 1e-3#np.sqrt(self.d)*np.exp(log_prior_cube_volume/self.d)
        L    = 20#np.sqrt(self.d)*log_prior_cube_volume/self.d
        print('Set up initial time step = {0}'.format(dt))
        print('Set up initial trajectory length = {0}'.format(L))
        return dt, L
        
    def unit_normal(self, q):
        """
        Returns the unit normal to the iso-Likelihood surface
        at x.
        
        It uses jax.grad for automatic differentiation, but reverts to
        finite differencing whenever the auto diff fails or the
        package is not found.
        
        Parameters
        ----------
        q : :obj:`cpnest.parameter.LivePoint`
            position
        
        Returns
        ----------
        n: :obj:`numpy.ndarray` unit normal to the logLmin contour evaluated at q
        """
        if self.finite_differencing == False:
            v    = self.constraint({n:q[n] for n in q.names})
            norm = np.sqrt(np.sum([v[vi]**2 for vi in v]))
            for vi in v: v[vi]/=norm
            normal_vector = np.array([v[vi] for vi in v])
        else:
            v    = self.constraint(q)
            normal_vector = v/np.linalg.norm(v)
        
        return normal_vector

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
    
    def update_time_step(self, acceptance, initialised = False):
        """
        Update the time step according to the
        acceptance rate. Correct also the trajectory length
        accordingly.
        
        Parameters
        ----------
        acceptance : :obj:'numpy.float'
        """
        if initialised == False:  self.dt, _  = self.time_step.update(acceptance)
        elif initialised == True: _, self.dt  = self.time_step.update(acceptance)

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
        return 0.5 * np.dot(p,np.dot(self.inverse_mass_matrix,p))-self.logdeterminant-0.5*self.d*np.log(2.0*np.pi)

    def hamiltonian(self, p, q):
        """
        Hamiltonian.
        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            momentum
        q : :obj:`cpnest.parameter.LivePoint`
            position
        Returns
        ----------
        H: :float: hamiltonian
        """
        return self.T(p) + self.V(q)

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
        initial_energy = self.hamiltonian(p0,q0)
        # evolve along the trajectory
        q, p = self.evolve_trajectory(p0, q0, *args)
        # minus sign from the definition of the potential
        final_energy   = self.hamiltonian(p,q)
        self.log_J = min(0.0, initial_energy-final_energy)
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
        p = p0 - 0.5 * self.dt * self.gradient(q0)
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

    def evolve_trajectory_one_step_position(self, p, q):
        """
        One leap frog step in position
        
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
        for j,k in enumerate(q.names):
            u, l  = self.prior_bounds[j][1], self.prior_bounds[j][0]
            q[k]  += self.dt * p[j] * self.inverse_mass[j]
            # check and reflect against the bounds
            # of the allowed parameter range
            while q[k] < l or q[k] > u:
                if q[k] > u:
                    q[k] = u - (q[k] - u)
                    p[j] *= -1
                if q[k] < l:
                    q[k] = l + (l - q[k])
                    p[j] *= -1
        return p, q

    def evolve_trajectory_one_step_momentum(self, p, q, logLmin, half = False):
        """
        One leap frog step in momentum
        
        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`cpnest.parameter.LivePoint`
            position
        logLmin: :obj:`numpy.float64`
            loglikelihood constraint
        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`cpnest.parameter.LivePoint` position
        """
        reflected = 0
        dV        = self.gradient(q)
        if half is True:
            factor = 0.5
        else:
            factor = 1.0

        c = self.check_constraint(q, logLmin)
        if c > 0:
            p += - factor * self.dt * dV
        else:
            normal = self.unit_normal(q)
            p += - 2.0*np.dot(p,normal)*normal
            reflected = 1
                
        return p, q, reflected

    def check_constraint(self, q, logLmin):
        """
        Check the likelihood
        
        Parameters
        ----------
        q0 : :obj:`cpnest.parameter.LivePoint`
        position
        logLmin: :obj:`numpy.float64`
        loglikelihood constraint
        Returns
        ----------
        c: :obj:`numpy.float64` value of the constraint
        """
        q.logP  = -self.V(q)
        q.logL  = self.log_likelihood(q)
        return q.logL - logLmin

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

        trajectory = [(q0,p0)]
        # evolve forward in time
        i = 0
        p, q, reflected = self.evolve_trajectory_one_step_momentum(p0.copy(), q0.copy(), logLmin, half = True)
        while (i < self.L):
            ll              = q.logL
            p, q            = self.evolve_trajectory_one_step_position(p, q)
            p, q, reflected = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = False)
            trajectory.append((q.copy(),p.copy()))
            i += 1
        
        p, q, reflected     = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = True)
        trajectory.append((q.copy(),p.copy()))
        # evolve backward in time
        i = 0
        p, q, reflected = self.evolve_trajectory_one_step_momentum(-p0.copy(), q0.copy(), logLmin, half = True)
        while (i < self.L):
            ll              = q.logL
            p, q            = self.evolve_trajectory_one_step_position(p, q)
            p, q, reflected = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = False)
            trajectory.append((q.copy(),p.copy()))
            i += 1

        p, q, reflected     = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = True)
        trajectory.append((q.copy(),p.copy()))
        
        if self.DEBUG: self.save_trajectory(trajectory, logLmin)
        return self.sample_trajectory(trajectory)

    def sample_trajectory(self, trajectory):
        """
        
        """
        logw = np.array([-self.hamiltonian(p,q) for q,p in trajectory[1:-1]])
        norm = logsumexp(logw)
        idx  = np.random.choice(range(1,len(trajectory)-1), p = np.exp(logw  - norm))
        return trajectory[idx]
                    
    def save_trajectory(self, trajectory, logLmin, filename = None):
        """
        save trajectory for diagnostic purposes
        """
        if filename is None:
            filename = 'trajectory_'+str(next(self.c))+'.txt'
        f = open(filename,'w')
        names = trajectory[0][0].names

        for n in names:
            f.write(n+'\t'+'p_'+n+'\t')
        f.write('logPrior\tlogL\tlogLmin\n')

        for j,step in enumerate(trajectory):
            q = step[0]
            p = step[1]
            for j,n in enumerate(names):
                f.write(repr(q[n])+'\t'+repr(p[j])+'\t')
            f.write(repr(q.logP)+'\t'+repr(q.logL)+'\t'+repr(logLmin)+'\n')
        f.close()

class DualAveragingStepSize:
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept):
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept

        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)

        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa

        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step

        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return np.exp(log_step), np.exp(self.log_averaged_step)
