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
from scipy.special import logsumexp
from .nest2pos import acl
from .parameter import LivePoint
import ray

try:
    from smt.surrogate_models import RBF
    no_smt = False
except:
    no_smt = True

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
        self.ensemble = ensemble
        for p in self.proposals:
            if isinstance(p,EnsembleProposal):
                p.set_ensemble(self.ensemble)

    def add_proposal(self, proposal, weight):
        self.proposals = self.proposals + [proposal]
        self.weights = self.weights + [weight]
        self.set_cycle()

class EnsembleSlice(EnsembleProposal):
    """
    The Ensemble Slice proposal from Karamanis & Beutler
    https://arxiv.org/pdf/2002.06212v1.pdf
    """
    log_J      = 0.0 # Symmetric proposal
    mean       = None
    covariance = None

    def set_ensemble(self,ensemble):
        """
        Over-ride default set_ensemble so that the
        mean and covariance matrix are recomputed when it is updated
        """
        super(EnsembleSlice,self).set_ensemble(ensemble)
        self.mean, self.covariance = ray.get(ensemble.get_mean_covariance.remote())

class EnsembleSliceDifferential(EnsembleSlice):
    """
    The Ensemble Slice Differential move from Karamanis & Beutler
    https://arxiv.org/pdf/2002.06212v1.pdf
    """

    def get_direction(self, mu = 1.0):
        """
        Draws two random points and returns their direction
        """
        subset = ray.get(self.ensemble.sample.remote(2))
        direction = reduce(LivePoint.__sub__,subset)
        return direction * mu

class EnsembleSliceCorrelatedGaussian(EnsembleSlice):
    """
    The Ensemble Slice Correlated Gaussian move from Karamanis & Beutler
    https://arxiv.org/pdf/2002.06212v1.pdf
    """
    def get_direction(self, mu = 1.0):
        """
        Draws a random gaussian direction
        """
        direction = mu * np.random.multivariate_normal(self.mean, self.covariance)
        return direction

class EnsembleSliceGaussian(EnsembleSlice):
    """
    The Ensemble Slice Gaussian move from Karamanis & Beutler
    https://arxiv.org/pdf/2002.06212v1.pdf
    """

    def get_direction(self, mu = 1.0):
        """
        Draw a random gaussian direction
        """
        direction  = np.random.normal(0.0,1.0,size=len(self.mean))
        direction /= np.linalg.norm(direction)
        return direction * mu

class EnsembleSliceProposalCycle(ProposalCycle):
    def __init__(self, model=None):
        """
        A proposal cycle that uses the slice sampler :obj:`EnsembleSlice`
        proposal.
        """
        weights = [1,1,1]
        proposals = [EnsembleSliceDifferential(),EnsembleSliceGaussian(),EnsembleSliceCorrelatedGaussian()]
        super(EnsembleSliceProposalCycle,self).__init__(proposals, weights)

    def get_direction(self, mu = 1.0, **kwargs):
        """
        Get a direction for the slice jump
        """
        self.idx = (self.idx + 1) % self.N
        p = self.cycle[self.idx]
        new = p.get_direction(mu = mu, **kwargs)
        self.log_J = p.log_J
        return new

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
        subset = ray.get(self.ensemble.sample.remote(self.Npoints))
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
        a = ray.get(self.ensemble.sample.remote(1))[0]
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
        a, b = ray.get(self.ensemble.sample.remote(2))
        sigma = 1e-4 # scatter around difference vector by this factor
        out = old + (b-a)*gauss(1.0,sigma)
        return out

class EnsembleEigenVector(EnsembleProposal):
    """
    A jump along a randomly-chosen eigenvector
    of the covariance matrix of the ensemble
    """
    log_J         = 0.0
    eigen_values  = None
    eigen_vectors = None
    covariance    = None
    ensemble      = None
    def set_ensemble(self, ensemble):
        """
        Over-ride default set_ensemble so that the
        eigenvectors are recomputed when it is updated
        """
        self.ensemble = ensemble
        self.update_eigenvectors()

    def update_eigenvectors(self):
        """
        Recompute the eigenvectors and eigevalues
        of the covariance matrix of the ensemble
        """
        self.eigen_values, self.eigen_vectors = ray.get(self.ensemble.get_eigen_quantities.remote())

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
        out = old.copy()
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
        weights = [5,
                   5,
                   10,
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
    covariance           = None
    mass_matrix          = None
    inverse_mass_matrix  = None
    momenta_distribution = None

    def __init__(self, model=None, **kwargs):
        """
        Initialises the class with the kinetic
        energy and the :obj:`cpnest.Model.potential`.
        """
        super(HamiltonianProposal, self).__init__(**kwargs)
        self.T                      = self.kinetic_energy
        self.V                      = model.potential
        self.dV                     = model.force
        self.prior_bounds           = model.bounds
        self.dimension              = len(self.prior_bounds)
        self.analytical_gradient    = model.analytical_gradient
        self.likelihood_gradient    = None
        self.dt                     = 0.1
        self.leaps                  = 100
        self.maxleaps               = 1000
        self.DEBUG                  = 0
        self.likelihood_gradient    = None
        self.initialised            = False
        self.TARGET                 = 0.65
        self.ADAPTATIONSIZE         = 0.001
        self.trajectories           = []

        self.set_mass_parameters()
        self.set_momenta_distribution()
        self._ensemble_initialised  = False

        if no_smt == True:
            print("ERROR! Current likelihood gradient approximation requires smt")
            exit()

    def set_mass_parameters(self):
        x = np.array([1 for _ in self.prior_bounds])# = np.array([p[1]-p[0] for p in self.prior_bounds])
        self.mass_matrix = np.diagflat(np.sqrt(x))
        self.inverse_mass_matrix         = np.linalg.inv(self.mass_matrix)
        self.inverse_mass        = np.atleast_1d(np.squeeze(np.diag(self.inverse_mass_matrix)))
        _, self.logdeterminant   = np.linalg.slogdet(self.mass_matrix)

    def set_ensemble(self, ensemble):
        """
        override the set ensemble method
        to update masses, momenta distribution
        and to heuristically estimate the normal vector to the
        hard boundary defined by logLmin.
        """
        if self._ensemble_initialised == False:
            self.ensemble = ensemble
            self.set_integration_parameters()
            self._ensemble_initialised = True

        if self.analytical_gradient == None:
            self.update_normal_vector()
            self.unit_normal = self.approximate_unit_normal
        else:
            self.likelihood_gradient = self.analytical_gradient
            self.unit_normal = self.exact_unit_normal

    def update_normal_vector(self):
        """
        update the constraint by approximating the
        loglikelihood hypersurface as a spline in
        each dimension.
        This is an approximation which
        improves as the algorithm proceeds
        """
        self.likelihood_gradient = ray.get(self.ensemble.get_likelihood_gradient.remote())

    def exact_unit_normal(self, q):
        v = self.likelihood_gradient(q)
        return v/np.linalg.norm(v)

    def approximate_unit_normal(self, q):
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

    def set_momenta_distribution(self):
        """
        update the momenta distribution using the
        mass matrix (precision matrix of the ensemble).
        """
        self.momenta_distribution = multivariate_normal(cov=np.identity(self.dimension))

    def set_integration_parameters(self):
        """
        Set the integration length according to the N-dimensional ellipsoid
        shortest and longest principal axes. The former sets to base time step
        while the latter sets the trajectory length
        """
        w, _                = ray.get(self.ensemble.get_eigen_quantities.remote())
        self.leaps          = int(np.ceil(w[-1]))
        self.max_dt         = 2.0*w[0]
        self.min_dt         = 1e-3
        self.dt             = np.sqrt(w[0]/self.leaps)

    def update_time_step(self, acceptance):
        """
        Update the time step according to the
        acceptance rate
        Parameters
        ----------
        acceptance : :obj:'numpy.float'
        """
        diff = acceptance - self.TARGET
        new_log_dt = np.log(self.dt) + self.ADAPTATIONSIZE * diff
        self.dt = np.minimum(np.exp(new_log_dt),self.max_dt)
        self.dt = np.maximum(self.min_dt,self.dt)

    def update_trajectory_length(self, safety = 20):
        """
        Update the trajectory length according to the estimated ACL
        Parameters
        ----------
        nmcmc :`obj`:: int
        """
        self.L = self.base_L + np.random.randint(nmcmc,5*nmcmc)

        ACL = np.array(ACL)
        # average over all trajectories and take the maximum over the dimensions
        self.leaps = int(np.max(np.average(ACL,axis=0)))

        if self.leaps < safety:
            self.leaps = safety

        if self.leaps > self.maxleaps:
            self.leaps = self.maxleaps

        self.trajectories = []

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
        q, p, r = self.evolve_trajectory(p0, q0, *args)
        # minus sign from the definition of the potential
        final_energy   = self.hamiltonian(p, q)
        if r == 1:
            self.log_J = -np.inf
        else:
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

        return q, -p, 0

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

    def counter(self):
        n = 0
        while True:
            yield n
            n += 1

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
            p += - 0.5 * self.dt * dV
            return p, q, reflected
        else:
            c = self.check_constraint(q, logLmin)
            if c > 0:
                p += - self.dt * dV
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
        trajectory = [(q0,p0,0)]
        # evolve forward in time
        i = 0
        p, q, reflected = self.evolve_trajectory_one_step_momentum(p0.copy(), q0.copy(), logLmin, half = True)
        while (i < self.leaps//2):
            p, q            = self.evolve_trajectory_one_step_position(p, q)
            p, q, reflected = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = False)
            trajectory.append((q.copy(),p.copy(),reflected))
            i += 1

        p, q, reflected     = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = True)
        if self.DEBUG: self.save_trajectory(trajectory, logLmin)

        # evolve backward in time
        i = 0
        p, q, reflected = self.evolve_trajectory_one_step_momentum(-p0.copy(), q0.copy(), logLmin, half = True)
        while (i < self.leaps//2):
            p, q            = self.evolve_trajectory_one_step_position(p, q)
            p, q, reflected = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = False)
            trajectory.append((q.copy(),p.copy(),reflected))
            i += 1

        p, q, reflected     = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = True)

        if self.DEBUG: self.save_trajectory(trajectory, logLmin)
        q, p, reflected = self.sample_trajectory(trajectory)

        self.trajectories.append(trajectory)
        return q, -p, reflected

    def sample_trajectory(self, trajectory):
        """

        """
        logw = np.array([-self.hamiltonian(p,q) for q,p,_ in trajectory[1:-1]])
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
        if self.c == 3: exit()
