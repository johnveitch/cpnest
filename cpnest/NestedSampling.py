from __future__ import division, print_function
import sys
import os
import pickle
import time
import logging
import numpy as np
from numpy import logaddexp
from numpy import inf
from math import isnan
from . import nest2pos
from .nest2pos import logsubexp
from operator import attrgetter
from .cpnest import CheckPoint
import functools
import ray
from tqdm import tqdm

def func(args):
    f, p, l = args
    return f(p, l)

logger = logging.getLogger('cpnest.NestedSampling')


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

    manager: `multiprocessing` manager instance which controls
        the shared objects.
        Default: None

    Nlive: int
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
                 n_periodic_checkpoint = None
                 ):
        """
        Initialise all necessary arguments and
        variables for the algorithm
        """
        loggername = 'cpnest.NestedSampling.NestedSampler'
        self.logger         = logging.getLogger(loggername)
        self.periodic_checkpoint_interval = periodic_checkpoint_interval
        self.model          = model
        self.nthreads       = nthreads
        self.prior_sampling = prior_sampling
        self.setup_random_seed(seed)
        self.verbose        = verbose
        self.acceptance     = 1.0
        self.accepted       = 0
        self.rejected       = 1
        self.queue_counter  = 0
        self.nlive          = nlive
        self.live_points    = [None] * self.nlive
        self.last_checkpoint_time = time.time()
        self.tolerance      = stopping
        self.condition      = np.inf
        self.worst          = 0
        self.logLmin        = -np.inf
        self.logLmax        = -np.inf
        self.iteration      = 0
        self.nested_samples = []
        self.logZ           = None
        self.state          = _NSintegralState(self.nlive)
        sys.stdout.flush()
        self.output_folder  = output
        self.output_file,self.evidence_file,self.resume_file = self.setup_output(output)
        header              = open(os.path.join(output,'header.txt'),'w')
        header.write('\t'.join(self.model.names))
        header.write('\tlogL\n')
        header.close()
        self.initialised    = False

    def setup_output(self,output):
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
        chain_filename = "chain_"+str(self.nlive)+"_"+str(self.seed)+".txt"
        output_file   = os.path.join(output,chain_filename)
        evidence_file = os.path.join(output,chain_filename+"_evidence.txt")
        resume_file  = os.path.join(output,"nested_sampler_resume.pkl")

        return output_file, evidence_file, resume_file


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
            f.write('{0:.5f} {1:.5f} {2:.2f}\n'.format(self.state.logZ, self.logLmax, self.state.info))

    def setup_random_seed(self,seed):
        """
        initialise the random seed
        """
        self.seed = seed
        np.random.seed(seed=self.seed)

    def consume_sample(self, pool):
        """
        consumes a sample from the consumer_pipes
        and updates the evidence logZ
        """
        # Increment the state of the evidence integration
        nreplace = len(self.manager.consumer_pipes)
        logLmin = self.get_worst_n_live_points(nreplace)
        self.state.increment(self.params[nreplace-1].logL, nreplace=nreplace)
        self.nested_samples.extend(self.params[:nreplace])
        logLtmp=[p.logL for p in self.params[:nreplace]]

        self.condition = logaddexp(self.state.logZ,self.logLmax - self.iteration/(float(self.nlive))) - self.state.logZ

        # make sure we are excluding the points to replace from the ensemble
        for s in pool.map_unordered(lambda a, v: a.set_ensemble.remote(self.live_points[self.nthreads:]), range(self.nthreads)): s
        # Replace the points we just consumed with the next acceptable ones
        while len(self.worst) > 0:

            indeces_to_remove = []
            p = pool.map(lambda a, v: a.produce_sample.remote(self.live_points[v], self.logLmin), self.worst)

            for i, r in zip(self.worst,p):

                acceptance,sub_acceptance,self.jumps,proposed = r

                if proposed.logL > self.logLmin:
                    # replace worst point with new one
                    self.live_points[i] = proposed.copy()
                    self.accepted += 1
                    indeces_to_remove.append(i)
                    self.iteration += 1

                    if self.verbose:

                        self.logger.info("{0:d}: n:{1:4d} NS_acc:{2:.3f} S{3:02d}_acc:{4:.3f} sub_acc:{5:.3f} H: {6:.2f} logL {7:.5f} --> {8:.5f} dZ: {9:.3f} logZ: {10:.3f} logLmax: {11:.2f}"\
                            .format(self.iteration, self.jumps, self.acceptance, self.iteration%self.nthreads, acceptance, sub_acceptance, self.state.info,\
                            logLtmp[i], self.live_points[i].logL, self.condition, self.state.logZ, self.logLmax))
                else:
                    self.rejected += 1

                self.acceptance = float(self.accepted)/float(self.accepted + self.rejected)

            for k in indeces_to_remove:
                self.worst.remove(k)

    def get_worst_n_live_points(self, n):
        """
        selects the lowest likelihood N live points
        for evolution
        """
        self.live_points.sort(key=attrgetter('logL'))
        self.worst = list(np.arange(n))
        self.logLmin = np.float128(self.live_points[n-1].logL)
        self.logLmax = np.float128(self.live_points[-1].logL)

        return np.float128(self.logLmin)

    def reset(self, pool):
        """
        Initialise the pool of `cpnest.parameter.LivePoint` by
        sampling them from the `cpnest.model.log_prior` distribution
        """
        nthreads=self.nthreads

        for i in range(self.nlive):
            self.live_points[i]      = self.model.new_point()
            self.live_points[i].logP = self.model.log_prior(self.live_points[i])
            self.live_points[i].logL = self.model.log_likelihood(self.live_points[i])

        # set up  the ensemble statistics
        for s in pool.map_unordered(lambda a, v: a.set_ensemble.remote(self.live_points), range(self.nthreads)): s
        p = pool.map(lambda a, v: a.produce_sample.remote(self.live_points[v], -np.inf), range(self.nlive))
        # send all live points to the samplers for start

        i = 0
        with tqdm(total=self.nlive, disable= not self.verbose, desc='CPNEST: populate samplers', position=nthreads) as pbar:
            for j,r in enumerate(p):
                acceptance,sub_acceptance,self.jumps,self.live_points[j] = r
                if np.isnan(self.live_points[j].logL):
                    self.logger.warn("Likelihood function returned NaN for params "+str(self.params))
                    self.logger.warn("You may want to check your likelihood function")
                if self.live_points[j].logP!=-np.inf and self.live_points[j].logL!=-np.inf:
                    pbar.update()

        # set up  the ensemble statistics
        for s in pool.map_unordered(lambda a, v: a.set_ensemble.remote(self.live_points), range(self.nthreads)): s
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

        if self.prior_sampling:
            for i in range(self.nlive):
                self.nested_samples.append(self.live_points[i])
            self.write_chain_to_file()
            self.write_evidence_to_file()
            self.logLmin = np.inf
            self.logLmax = np.inf
            self.logger.warning("Nested Sampling process {0!s}, exiting".format(os.getpid()))
            return 0

        try:
            while self.condition > self.tolerance:
                self.consume_sample(pool)
                if time.time() - self.last_checkpoint_time > self.periodic_checkpoint_interval:
                    self.checkpoint()
                    self.last_checkpoint_time = time.time()
        except CheckPoint:
            self.checkpoint()
            sys.exit(130)

        # Signal worker threads to exit
        self.logLmin = np.inf

        # final adjustments
        self.live_points.sort(key=attrgetter('logL'))
        for i,p in enumerate(self.live_points):
            self.state.increment(p.logL,nlive=self.nlive-i)
            self.nested_samples.append(p)

        # Refine evidence estimate
        self.state.finalise()
        self.logZ = self.state.logZ
        # output the chain and evidence
        self.write_chain_to_file()
        self.write_evidence_to_file()
        self.logger.critical('Final evidence: {0:0.2f}'.format(self.state.logZ))
        self.logger.critical('Information: {0:.2f}'.format(self.state.info))

        # Some diagnostics
        if self.verbose>1 :
          self.state.plot(os.path.join(self.output_folder,'logXlogL.png'))
        return self.state.logZ, self.nested_samples

    def checkpoint(self):
        """
        Checkpoint its internal state
        """
        self.logger.critical('Checkpointing nested sampling')
        with open(self.resume_file,"wb") as f:
            pickle.dump(self, f)

    @classmethod
    def resume(cls, filename, manager, usermodel):
        """
        Resumes the interrupted state from a
        checkpoint pickle file.
        """
        with open(filename,"rb") as f:
            obj = pickle.load(f)
        obj.manager = manager
        obj.logLmin = obj.llmin
        obj.logLmax = obj.llmax
        obj.model = usermodel
        obj.logger = logging.getLogger("cpnest.NestedSampling.NestedSampler")
        del obj.__dict__['llmin']
        del obj.__dict__['llmax']
        obj.logger.critical('Resuming NestedSampler from ' + filename)
        obj.last_checkpoint_time = time.time()
        return obj

    def __getstate__(self):
        state = self.__dict__.copy()
        state['llmin'] = self.logLmin
        state['llmax'] = self.logLmax
        # Remove the unpicklable entries.
        del state['logLmin']
        del state['logLmax']
        del state['manager']
        del state['model']
        del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
