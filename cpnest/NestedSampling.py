from __future__ import division, print_function
import sys
import os
import dill
import time
import logging
import bisect
import numpy as np
from numpy import logaddexp
from numpy import inf
from scipy.stats import kstest
from math import isnan
from . import nest2pos
from .nest2pos import logsubexp
from operator import attrgetter
from .cpnest import CheckPoint
from .utils import auto_garbage_collect
import copy

import ray
import random
from tqdm import tqdm
logger = logging.getLogger('cpnest.NestedSampling')

class KeyOrderedList(list):
    """
    List object that is ordered according to a key

    Parameters
    ----------
    iterable : array_like
        Initial input used to intialise the list
    key : function, optional
        Key to use to sort the list, by defaul it is sorted by its
        values.
    """
    def __init__(self, iterable, key=lambda x: x):
        iterable = sorted(iterable, key=key)
        super(KeyOrderedList, self).__init__(iterable)

        self._key = key
        self._keys = [self._key(v) for v in iterable]

    def search(self, item):
        """
        Find the location of a new entry
        """
        return bisect.bisect(self._keys, self._key(item))

    def add(self, item):
        """
        Update the ordered list with a single item and return the index
        """
        index = self.search(item)
        self.insert(index, item)
        self._keys.insert(index, self._key(item))
        return index


class OrderedLivePoints(KeyOrderedList):
    """
    Object that contains live points ordered by increasing log-likelihood. Requires
    the log-likelihood to be pre-computed.

    Assumes the log-likelihood is accesible as an attribute of each live point.

    Parameters
    ----------
    live_points : array_like
        Initial live points
    """
    def __init__(self, live_points):
        super(OrderedLivePoints, self).__init__(live_points, key=lambda x: x.logL)

    def insert_live_point(self, live_point):
        """
        Insert a live point and return the index of the new point
        """
        return self.add(live_point)

    def remove_n_worst_points(self, n):
        """
        Remove the n worst live points
        """
        del self[:n]
        del self._keys[:n]


class _NSintegralState(object):
    """
    Stores the state of the nested sampling integrator
    """
    def __init__(self, nlive):
        self.nlive = nlive
        self.reset()
        loggername = 'cpnest.NestedSampling._NSintegralState'
        self.logger = logging.getLogger(loggername)

    def reset(self):
        """
        Reset the sampler to its initial state at logZ = -infinity
        """
        self.iteration = 0
        self.logZ = -inf
        self.oldZ = -inf
        self.logw = 0
        self.info = 0
        # Start with a dummy sample enclosing the whole prior
        self.logLs = [-inf]  # Likelihoods sampled
        self.log_vols = [0.0]  # Volumes enclosed by contours

    def increment(self, logL, nlive=None, nreplace=1):
        """
        Increment the state of the evidence integrator
        Simply uses rectangle rule for initial estimate
        see: https://www.cell.com/biophysj/pdf/S0006-3495(12)00055-0.pdf
        for parallel implementation
        """
        if(logL<=self.logLs[-1]):
            self.logger.warning('NS integrator received non-monotonic logL. {0:.5f} -> {1:.5f}'.format(self.logLs[-1], logL))
        if nlive is None:
            nlive = self.nlive
        oldZ = self.logZ
        logt=-nreplace/nlive
        Wt = self.logw + logL + logsubexp(0,logt)
        self.logZ = logaddexp(self.logZ,Wt)
        # Update information estimate
        if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
            self.info = np.exp(Wt - self.logZ)*logL + np.exp(oldZ - self.logZ)*(self.info + oldZ) - self.logZ
            if isnan(self.info):
                self.info = 0

        # Update history
        self.logw += logt
        self.iteration += 1
        self.logLs.append(logL)
        self.log_vols.append(self.logw)

    def finalise(self):
        """
        Compute the final evidence with more accurate integrator
        Call at end of sampling run to refine estimate
        """
        from scipy import integrate
        # Trapezoidal rule
        self.logZ = nest2pos.log_integrate_log_trap(np.array(self.logLs), np.array(self.log_vols))
        return self.logZ

    def plot(self,filename):
        """
        Plot the logX vs logL
        """
        import matplotlib as mpl
        mpl.use('Agg')
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.log_vols,self.logLs)
        plt.title('{0} iterations. logZ={1:.2f} H={2:.2f} bits'.format(self.iteration,self.logZ,self.info*np.log2(np.e)))
        plt.grid(which='both')
        plt.xlabel('log prior_volume')
        plt.ylabel('log likelihood')
        plt.xlim([self.log_vols[-1],self.log_vols[0]])
        plt.savefig(filename)
        self.logger.info('Saved nested sampling plot as {0}'.format(filename))

    def __getstate__(self):
        """Remove the unpicklable entries."""
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        if 'logger' not in state:
            loggername = "cpnest.NestedSampling._NSintegralState"
            state['logger'] = logging.getLogger(loggername)
        self.__dict__ = state

class NestedSampler(object):
    """
    Nested Sampler class.
    Initialisation arguments:

    model: :obj:`cpnest.Model` user defined model

    periodic_checkpoint_interval: float
        time interval in seconds between periodic checkpoints

    nthreads: int
        number of parallel sampling threads

    nlive: int
        number of live points to be used for the integration
        Default: 1024

    output: string
        folder where the output will be stored
        Default: None

    verbose: int
        0: Nothing
        1: display information on screen
        2: (1) + diagnostic plots
        Default: 1

    seed: int
        seed for the initialisation of the pseudorandom chain
        Default: 1234

    prior_sampling: boolean
        produce Nlive samples from the prior.
        Default: False

    stopping: float
        Stop when remaining samples wouldn't change logZ estimate by this much.
        Deafult: 0.1

    n_periodic_checkpoint: int
        **deprecated**
        This parameter should not be used, it should be set by the manager instead.
        checkpoint the sampler every n_periodic_checkpoint iterations
        Default: None (disabled)

    resume_file: string
        file where the checkpoints will be stored
        Default: None

    state: dict
        dictionary holding the checkpointed state
        Default: None
    """

    def __init__(self,
                 model,
                 periodic_checkpoint_interval = np.inf,
                 nthreads       = None,
                 nlive          = 1024,
                 output         = None,
                 verbose        = 1,
                 seed           = 1,
                 prior_sampling = False,
                 stopping       = 0.1,
                 n_periodic_checkpoint = None,
                 position       = 0,
                 resume_file    = None,
                 state          = None
                 ):
        """
        Initialise all necessary arguments and
        variables for the algorithm
        """
        loggername          = 'cpnest.NestedSampling.NestedSampler'
        self.logger         = logging.getLogger(loggername)
        self.logger.addHandler(logging.StreamHandler())
        self.model          = model

        if state is None:
            self.position       = position
            self.periodic_checkpoint_interval = periodic_checkpoint_interval
            self.nthreads       = nthreads
            self.prior_sampling = prior_sampling
            self.setup_random_seed(seed)
            self.verbose        = verbose
            self.acceptance     = 1.0
            self.accepted       = 0
            self.rejected       = 1
            self.queue_counter  = 0
            self.nlive          = nlive
            self.live_points    = None
            self.last_checkpoint_time = time.time()
            self.tolerance      = stopping
            self.condition      = np.inf
            self.worst          = 0
            self.logLmin        = -np.inf
            self.logLmax        = -np.inf
            self.iteration      = 0
            self.prior_samples  = []
            self.nested_samples = []
            self.rolling_p      = []
            self.logZ           = None
            self.resume_file    = resume_file
            sys.stdout.flush()
            self.output_folder  = output
            self.output_file,self.evidence_file = self.setup_output(output)
            header              = open(os.path.join(output,'header.txt'),'w')
            header.write('\t'.join(self.model.names))
            header.write('\tlogL\n')
            header.close()
            self.initialise_live_points()
            self.initialised    = False
        else:
            for k,v in state.items():
                setattr(self,k,v)
            # silly workaround for extra live points if caught in the middle of
            # updating them
            if self.nlive != len(self.live_points._list):
                self.live_points._list.pop(-1)

    def initialise_live_points(self):

        l = []

        with tqdm(total=self.nlive, disable = not self.verbose, desc='CPNEST: populate samplers', position=self.position) as pbar:
            for i in range(self.nlive):

                p = self.model.new_point()
                p.logP = self.model.log_prior(p)
                p.logL = self.model.log_likelihood(p)
                l.append(p)
                pbar.update()

        self.live_points = LivePoints(l, self.nthreads)

    def setup_output(self, output):
        """
        Set up the output folder

        -----------
        Parameters:
        output: string
            folder where the results will be stored
        -----------
        Returns:
            output_file, evidence_file, resume_file: tuple
                output_file:   file where the nested samples will be written
                evidence_file: file where the evidence will be written
                resume_file:   file used for checkpointing the algorithm
        """
        chain_filename = "chain_"+str(self.nlive)+"_"+str(self.position)+".txt"
        output_file   = os.path.join(output,chain_filename)
        evidence_file = os.path.join(output,chain_filename+"_evidence_"+str(self.position)+".txt")

        return output_file, evidence_file


    def write_chain_to_file(self):
        """
        Outputs a `cpnest.parameter.LivePoint` to the
        output_file
        """
        with open(self.output_file,"w") as f:
            f.write('{0:s}\n'.format(self.model.header().rstrip()))
            for ns in self.nested_samples:
                f.write('{0:s}\n'.format(self.model.strsample(ns).rstrip()))

    def write_evidence_to_file(self):
        """
        Write the evidence logZ and maximum likelihood to the evidence_file
        """
        with open(self.evidence_file,"w") as f:

            f.write('#logZ\tlogLmax\tH\n')
            f.write('{0:.5f} {1:.5f} {2:.2f}\n'.format(self.live_points.get_logZ(),
                                                       self.live_points.get_logLmax(),
                                                       self.live_points.get_info()))

    def setup_random_seed(self,seed):
        """
        initialise the random seed
        """
        self.seed = seed
        np.random.seed(seed=self.seed)

    def consume_sample(self, pool):
        """
        requests the workers to update the live points
        and updates the evidence logZ
        """
        # Get the worst live points
        self.worst   = self.live_points.get_worst(self.nthreads)
        # get the necessary statistics from the LivePoint actor
        self.logLmin, self.logLmax, logZ, info = self.live_points.get_logLs_logZ_info()

        if self.verbose:
            logLtmp = self.live_points.get_worst_logLs()

        self.condition = logaddexp(logZ,self.logLmax - self.iteration/(float(self.nlive))) - logZ
        
        # set up the ensemble statistics
        if (self.iteration % self.nlive//10) < self.nthreads:
            lpp = LivePoints(self.live_points._list, self.nthreads)
            lp = ray.put(lpp)
            for s in pool.map_unordered(lambda a, v: a.set_ensemble.remote([lp]), range(self.nthreads)):
                pass
            
        starting_points = self.live_points.sample(self.nthreads)

        for v in starting_points:
            pool.submit(lambda a, v: a.produce_sample.remote(v, self.logLmin), v)

        i = 0
        while pool.has_next():

            acceptance, sub_acceptance, self.jumps, proposed = pool.get_next_unordered()

            if proposed.logL > self.logLmin:
                # replace worst point with new one
                self.live_points.insert(i%self.nthreads,proposed.copy())
                self.accepted  += 1
                self.iteration += 1

                if self.verbose:
                    self.logger.info("{0:d}: n:{1:4d} NS_acc:{2:.3f} S{3:02d}_acc:{4:.3f} sub_acc:{5:.3f} H: {6:.2f} logL {7:.5f} --> {8:.5f} dZ: {9:.3f} logZ: {10:.3f} logLmax: {11:.2f}"\
                        .format(self.iteration, self.jumps, self.acceptance, i%self.nthreads, acceptance, sub_acceptance, info,\
                        logLtmp[i%self.nthreads], proposed.logL, self.condition, logZ, self.logLmax))

                i += 1

            else:
                self.rejected += 1
                p = pool.submit(lambda a, v: a.produce_sample.remote(v, self.logLmin), starting_points[i%self.nthreads])

            self.acceptance = float(self.accepted)/float(self.accepted + self.rejected)

        self.live_points.remove_n_worst_points(self.nthreads)
        self.live_points.update_mean_covariance()

    def reset(self, pool):
        """
        Initialise the pool of `cpnest.parameter.LivePoint` by
        sampling them from the `cpnest.model.log_prior` distribution
        """
        # set up  the ensemble statistics
        lpp = LivePoints(self.live_points._list, self.nthreads)
        lp = ray.put(lpp)
        
        for s in pool.map_unordered(lambda a, v: a.set_ensemble.remote([lp]), range(self.nthreads)):
            pass
        
        # send all live points to the samplers for start
        for i in range(self.nlive):
            pool.submit(lambda a, v: a.produce_sample.remote(v, -np.inf), self.live_points.get(i))

        i = 0

        with tqdm(total=self.nlive, disable= not self.verbose, desc='CPNEST: sampling prior', position=self.position) as pbar:
            while pool.has_next():
                acceptance,sub_acceptance,self.jumps,x = pool.get_next()
                if np.isnan(x.logL):
                    self.logger.warning("Likelihood function returned NaN for params "+str(x))
                    self.logger.warning("You may want to check your likelihood function")
                if np.isfinite(x.logP) and np.isfinite(x.logL):
                    self.live_points.set(i,x)
                    self.prior_samples.append(x)
                    i += 1
                    pbar.update()
                else:
                    pool.submit(lambda a, v: a.produce_sample.remote(v, -np.inf), self.live_points.get(i))

        self.live_points.update_mean_covariance()
        self.live_points.set_ordered_list()

        if self.verbose:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self.initialised=True

    def nested_sampling_loop(self, pool):
        """
        main nested sampling loop
        """
        if not self.initialised:
            self.reset(pool)
        else:
            self.logger.info("Nested Sampling process {0!s}, restoring samplers".format(os.getpid()))
            lp_ref = ray.put(self.live_points)
            for s in pool.map_unordered(lambda a, v: a.set_ensemble.remote([lp_ref]), range(self.nthreads)):
                pass

        if self.prior_sampling:
            self.logZ, self.nested_samples = self.live_points.finalise()
            self.write_chain_to_file()
            self.write_evidence_to_file()
            self.logLmin = np.inf
            self.logLmax = np.inf
            self.logger.info("Nested Sampling process {0!s}, exiting".format(os.getpid()))
            return 0

        try:
            while self.condition > self.tolerance:
                self.consume_sample(pool)
                if time.time() - self.last_checkpoint_time > self.periodic_checkpoint_interval:
                    self.save()
                    self.last_checkpoint_time = time.time()

                if (self.iteration % self.nlive) < self.nthreads:
                    self.check_insertion_indices()
                auto_garbage_collect()

        except CheckPoint:
            self.checkpoint()
            sys.exit(130)

        # Refine evidence estimate
        self.logZ, self.nested_samples = self.live_points.finalise()
        self.info = self.live_points.get_info()
        # output the chain and evidence
        self.write_chain_to_file()
        self.write_evidence_to_file()
        if self.verbose:
            self.logger.critical('Final evidence: {0:0.2f}'.format(self.logZ))
            self.logger.critical('Information: {0:.2f}'.format(self.info))

        # Some diagnostics
        if self.verbose > 1 :
            self.live_points.plot(os.path.join(self.output_folder,'logXlogL.png'))

        return 0

    def check_insertion_indices(self, rolling=True, filename=None):
        """
        Checking the distibution of the insertion indices either during
        the nested sampling run (rolling=True) or for the whole run
        (rolling=False).
        """
        if not self.live_points.get_insertion_indices():
            return
        if rolling:
            indices = self.live_points.get_insertion_indices()[-self.nlive:]
        else:
            indices = self.live_points.get_insertion_indices()

        D, p = kstest(indices, 'uniform', args=(0, 1))
        if rolling:
            if self.verbose:
                self.logger.info('Rolling KS test: D={0:.3}, p-value={1:.3}'.format(D, p))
            self.rolling_p.append(p)
        else:
            if self.verbose:
                self.logger.info('Final KS test: D={0:.3}, p-value={1:.3}'.format(D, p))

        if filename is not None:
            np.savetxt(os.path.join(
                self.output_folder, filename),
                self.live_points.get_insertion_indices(),
                newline='\n',delimiter=' ')

    def get_output_folder(self):
        return self.output_folder

    def get_nested_samples(self):
        return self.nested_samples

    def get_prior_samples(self):
        return self.prior_samples

    def get_live_points(self):
        return self.live_points

    def get_logZ(self):
        return self.logZ

    def get_nlive(self):
        return self.nlive

    def get_information(self):
        return self.info

    def save(self):
        state = self.__dict__.copy()
        self.logger.critical('Saving Nested Sampling state in ' + str(self.resume_file))
        del state['model']
        del state['logger']

        with open(self.resume_file,'wb') as f:
            dill.dump(state,f)

class LivePoints:
    """
    class holding the live points pool

    parameters
    ==========

    l: list
        list of live points generated from the `cpnest.model` new_point method

    n_replace: int
        number of live points to be replaced at each step. must coincide with the number of
        sampling threads

    verbose: int
        level of verbosity
    """
    def __init__(self, l, n_replace = 1, verbose = 2):
        self.n_replace            = n_replace
        self._list                = l
        self.n                    = len(l)
        self.dim                  = self._list[0].dimension
        self.mean                 = None
        self.covariance           = None
        self.eigen_values         = None
        self.eigen_vectors        = None
        self.likelihood_gradient  = None
        self.worst                = None
        self.logLmax              = -np.inf
        self.logLmin              = -np.inf
        self.state                = _NSintegralState(self.n)
        self.logger               = logging.getLogger('CPNest')
        self.nested_samples       = []
        self.insertion_indices    = []
        self.update_mean_covariance()

    def set_ordered_list(self):
        """
        initialise the list of live points as a ordered list
        """
        self._list = OrderedLivePoints(self._list)

    def get(self, i):
        """
        return the i-th element
        """
        return self._list[i]

    def get_mean_covariance(self):
        """
        return the live points sample mean and covariance
        """
        return self.get_mean(), self.get_covariance()

    def get_covariance(self):
        """
        return the live points sample covariance
        """
        return self.covariance

    def get_mean(self):
        """
        return the live points sample mean
        """
        return self.mean

    def get_eigen_quantities(self):
        """
        return the live points sample covariance eigen values and eigen vectors
        """
        return self.eigen_values, self.eigen_vectors

    def get_length(self):
        """
        return the number of live points
        """
        return self.n

    def get_dimension(self):
        """
        return the dimension of each live point
        """
        return self.dim

    def set(self,i, val):
        """
        set the i-th live point to val
        """
        self._list[i] = val

    def insert(self, i, val):
        """
        insert val in the ordered list
        """
        index = self._list.insert_live_point(val)
        self.insertion_indices.append((index - self.n_replace) / (self.n - i - 1))

    def to_list(self):
        """
        return the list of live points
        """
        return self._list

    def update_mean_covariance(self):
        """
        Recompute mean and covariance matrix
        of the ensemble of Live points
        """
        cov_array = np.zeros((self.dim,self.n))
        if self.dim == 1:
            name=self._list[0].names[0]
            self.covariance = np.atleast_2d(np.var([self._list[j][name] for j in range(self.n)]))
            self.mean       = np.atleast_1d(np.mean([self._list[j][name] for j in range(self.n)]))
        else:
            for i,name in enumerate(self._list[0].names):
                for j in range(self.n): cov_array[i,j] = self._list[j][name]
            self.covariance = np.cov(cov_array)
            self.mean       = np.mean(cov_array,axis=1)
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(self.covariance)

    def get_as_array(self):
        """
        return the live points as a stacked (n,d) numpy array
        """
        n = self.n+len(self.nested_samples)
        as_array = np.zeros((self.dim,n))
        for i,name in enumerate(self._list[0].names):
            for j in range(self.n): as_array[i,j] = self._list[j][name]
            for j in range(len(self.nested_samples)): as_array[i,j+self.n] = self.nested_samples[j][name]

        return as_array.T # as an array (N,D)

    def sample(self, n):
        """
        randomly sample n live points
        """
        if self.worst == None:
            return random.sample(self._list, n)
        else:
            k = len(self.worst)
            return random.sample(self._list[k:], n)

    def sample_around(self, q, n):
        """
        sample n live points closest to q
        """
        distance = lambda list_value : np.sum(list_value.values - q.values)
        closest_value = min(self._list, key=distance)
        idx = self._list.index(closest_value)
        return [self.get(i) for i in range(n//2-idx,n//2+idx)]

    def get_logLmax(self):
        """
        return the maximum likelihood value
        """
        return self.logLmax

    def get_logLmin(self):
        """
        return the minimum likelihood value
        """
        return self.logLmin

    def get_worst(self, n):
        """
        selects the lowest likelihood N live points
        and updates the integral state
        """
        self.logLmin = self._list[n-1].logL
        self.logLmax = self._list[-1].logL
        self.worst = self._list[:n]
        self.nested_samples.extend(self.worst)
        self.state.increment(self.worst[n-1].logL, nreplace=self.n_replace)
        return self.worst

    def remove_n_worst_points(self,n):
        """
        remove the set of worst n live points
        """
        self._list.remove_n_worst_points(n)

    def get_insertion_indices(self):
        """
        return the insertion indeces
        """
        return self.insertion_indices

    def get_info(self):
        """
        return the information
        """
        return self.state.info

    def get_logZ(self):
        """
        return the log evidence
        """
        return self.state.logZ

    def get_logLs_logZ_info(self):
        return self.get_logLmin(), self.get_logLmax(), self.get_logZ(), self.get_info()

    def get_worst_logLs(self):
        return np.array([w.logL for w in self.worst])

    def finalise(self):
        # final adjustments
        self._list.sort(key=attrgetter('logL'))
        for i,p in enumerate(self._list):
            self.state.increment(p.logL, nlive=self.n-i, nreplace = 1)
            self.nested_samples.append(p)

        # Refine evidence estimate
        self.logZ = self.state.finalise()
        return self.state.logZ, self.nested_samples

    def plot(self,filename):
        self.state.plot(filename)
        return 0

    def _get_integral_state(self):
        return self.state

    def _set_internal_state(self, state):
        self.state = state
