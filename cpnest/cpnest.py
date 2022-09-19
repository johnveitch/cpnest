# coding: utf-8

import numpy as np
import os
import sys
import signal
import logging
import dill

from .utils import LogFile

# module logger takes name according to its path
LOGGER = logging.getLogger('cpnest.cpnest')
import ray
from ray.util import ActorPool

class CheckPoint(Exception):
    pass

def sighandler(signal, frame):
    LOGGER.critical("Handling signal {}".format(signal))
    raise CheckPoint()

class CPNest(object):
    """
    Class to control CPNest sampler
    cp = CPNest(usermodel,nlive=100,output='./',verbose=0,seed=None,maxmcmc=100,nthreads=None,balanced_sampling = True)

    Input variables:
    =====

    usermodel: :obj:`cpnest.Model`
        a user-defined model to analyse

    nlive: `int`
        Number of live points (100)

    output : `str`
        output directory (./)

    verbose: `int`
        Verbosity:
            0: no display of information on screen, save the NS chain and evidence
            1: 0 + display progress on screen
            2: 1 + display diagnostics (ACL), save the posterior samples and trace plots and posterior plots
            3: 2 + save chains from individual samplers
            default: 0

    seed: `int`
        random seed (default: 1234)

    maxmcmc: `int`
        maximum MCMC points for MHS sampling chains (5000)

    nnest: `int`
        number of parallel nested samplers.
        Specifying nnest > 1 will run multiple parallel nested samplers, and combine the
        results at the end to produce a single evidence and set of posterior samples.
        Default: 1

    nensemble: `int`
        number of sampler threads using an ensemble samplers. Default: 1

    nhamiltonian: `int`
        number of sampler threads using an hamiltonian samplers. Default: 0

    nslice: `int`
            number of sampler threads using an ensemble slice samplers. Default: 0

    resume: `boolean`
        determines whether cpnest will resume a run or run from scratch. Default: False.

    proposals: `dict`
        dictionary of lists with custom jump proposals.
        key 'mhs' for the Metropolis-Hastings sampler,
        'hmc' for the Hamiltonian Monte-Carlo sampler,
        'sli' for the slice sampler.
        Default: None
        By default, the sampler will use the  :obj:`cpnest.proposal.DefaultProposalCycle`
        for the mhs sampler; :obj:`cpnest.proposal.HamiltonianProposalCycle` for hmc and
        :obj:`cpnest.proposal.EnsembleSliceProposalCycle` for the sli sampler.

    prior_sampling: `boolean`
        Default: False
        If true, generates samples from the prior then terminates. Adjust `nlive` to control the number
        of samples requested.

    n_periodic_checkpoint: `int`
        **deprecated**
        checkpoint the sampler every n_periodic_checkpoint iterations
        Default: None (disabled)

    periodic_checkpoint_interval: `float`
        checkpoing the sampler every periodic_checkpoint_interval seconds
        Default: None (disabled)

    object_store_memory: `int`
        amount of memory in bytes reserved for ray object store
        Default: 1e9 (1GB)

    """
    def __init__(self,
                 usermodel,
                 nlive        = 100,
                 output       = './',
                 verbose      = 0,
                 seed         = None,
                 maxmcmc      = 5000,
                 nnest        = 1,
                 nensemble    = 1,
                 nhamiltonian = 0,
                 nslice       = 0,
                 resume       = False,
                 proposals     = None,
                 n_periodic_checkpoint = None,
                 periodic_checkpoint_interval=None,
                 prior_sampling = False,
                 object_store_memory=10**9,
                 poolsize=None,
                 nthreads=None
                 ):

        self.verbose   = verbose
        self.logger    = logging.getLogger('cpnest.cpnest.CPNest')
        self.nsamplers = nensemble+nhamiltonian+nslice
        self.nnest     = nnest
        if nthreads is not None and self.nsamplers == 0:
            nensemble = nthreads
            self.nsamplers = nensemble
            self.logger.warning(f'DEPRECATION WARNING: nthreads is deprecated. Defaulting to use nensemble={nthreads}')
        assert self.nsamplers > 0, "no sampler processes requested!"
        import psutil
        self.max_threads = psutil.cpu_count()

        if self.nsamplers%self.nnest != 0:
            self.logger.warning("Error! Number of samplers not balanced")
            self.logger.warning("to the number of nested samplers! Exiting.")
            exit(-1)

        self.nthreads = self.nsamplers+self.nnest

        if self.nthreads > self.max_threads:
            self.logger.warning("More cpus than available are being requested!")
            self.logger.warning("This might result in excessive overhead")

        self.ns_pool = []
        self.pool    = []

        if not ray.is_initialized():
            self.existing_cluster = False
            ray.init(num_cpus=self.nthreads,
                     ignore_reinit_error=True,
                     object_store_memory=object_store_memory)
        else:
            self.existing_cluster = True
            ray.init(address='auto',
                     num_cpus=self.nthreads,
                     ignore_reinit_error=True,
                     object_store_memory=object_store_memory)

        assert ray.is_initialized() == True
        output = os.path.join(output, '')
        checkpoint_folder = os.path.join(output,'checkpoints')
        os.makedirs(output, exist_ok=True)
        os.makedirs(checkpoint_folder, exist_ok=True)
        
        # Import placement group APIs.
        from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group

        # set up the resume files
        self.resume_file = []
        # The LogFile context manager ensures everything within is logged to
        # 'cpnest.log' but the file handler is safely closed once the run is
        # finished.
        self.log_file = LogFile(os.path.join(output, 'cpnest.log'),
                                verbose=self.verbose)

        with self.log_file:
            if poolsize is not None:
                self.logger.warning('poolsize is a deprecated option and will \
                                    be removed in a future version.')
            if nthreads is not None:
                self.logger.warning('nthreads is a deprecated option and will\
                                    be removed in a future verison.')
            if self.verbose:
                self.logger.info('Running with {0} parallel threads'.format(self.nthreads))
                self.logger.info('Nested samplers: {0}'.format(nnest))
                self.logger.info('Ensemble samplers: {0}'.format(nensemble))
                self.logger.info('Slice samplers: {0}'.format(nslice))
                self.logger.info('Hamiltonian samplers: {0}'.format(nhamiltonian))
                self.logger.info('ray object store size: {0} GB'.format(object_store_memory/1e9))

            if n_periodic_checkpoint is not None:
                self.logger.critical(
                    "The n_periodic_checkpoint kwarg is deprecated, "
                    "use periodic_checkpoint_interval instead."
                )
            if periodic_checkpoint_interval is None:
                self.periodic_checkpoint_interval = np.inf
            else:
                self.periodic_checkpoint_interval = periodic_checkpoint_interval

            from .sampler import HamiltonianMonteCarloSampler, MetropolisHastingsSampler, SliceSampler, SamplersCycle
            from .NestedSampling import NestedSampler
            from .proposal import DefaultProposalCycle, HamiltonianProposalCycle, EnsembleSliceProposalCycle
            if proposals is None:
                proposals = dict(mhs=DefaultProposalCycle,
                                 hmc=HamiltonianProposalCycle,
                                 sli=EnsembleSliceProposalCycle)
            elif type(proposals) == list:
                proposals = dict(mhs=proposals[0],
                                 hmc=proposals[1],
                                 sli=proposals[2])
            self.nlive    = nlive
            self.output   = output
            self.results  = {}
            self.prior_sampling = prior_sampling
            self.user     = usermodel
            self.resume = resume
            
            if seed is not None:
                self.seed = iter([np.random.SeedSequence(seed+i) for i in range(self.nthreads)])
            else:
                self.logger.warning('seed = None was passed. The results will not be reproducible. \
                                     Set seed = int to have a reproducible sequence.')
                self.seed = seed
            
            rng = None
            for j in range(self.nnest):
                
                if self.seed is not None:
                    s0 = next(self.seed)
                    rng = np.random.default_rng(s0)

                pg = placement_group([{"CPU": 1+self.nsamplers//self.nnest}],strategy="STRICT_PACK")
                ray.get(pg.ready())
                
                self.resume_file.append(os.path.join(checkpoint_folder, "nested_sampler_resume_{}.pkl".format(j)))
                
                if not os.path.exists(self.resume_file[j]) or resume == False:
                    
                    self.ns_pool.append(ray.remote(NestedSampler).options(placement_group=pg).remote(
                                self.user,
                                nthreads       = self.nsamplers,
                                nlive          = nlive,
                                output         = output,
                                verbose        = self.verbose,
                                rng            = rng,
                                prior_sampling = self.prior_sampling,
                                periodic_checkpoint_interval = self.periodic_checkpoint_interval,
                                resume_file    = self.resume_file[j],
                                position = j))
                else:
                    state = self.load_nested_sampler_state(self.resume_file[j])
                    ns = ray.remote(NestedSampler).options(placement_group=pg).remote(
                                self.user,
                                nthreads       = self.nsamplers,
                                nlive          = nlive,
                                output         = output,
                                verbose        = self.verbose,
                                prior_sampling = self.prior_sampling,
                                periodic_checkpoint_interval = self.periodic_checkpoint_interval,
                                resume_file    = self.resume_file[j],
                                position = j,
                                state    = state)

                    self.ns_pool.append(ns)
                
                samplers = []

                # instantiate the sampler class
                for i in range(nensemble//self.nnest):
                    
                    if self.seed is not None:
                        rng = np.random.default_rng(next(self.seed))
                    
                    S = MetropolisHastingsSampler.options(placement_group=pg).remote(self.user,
                                          maxmcmc,
                                          rng         = rng,
                                          verbose     = self.verbose,
                                          proposal    = proposals['mhs']
                                          )
                    samplers.append(S)
                
                for i in range(nhamiltonian//self.nnest):
                    
                    if self.seed is not None:
                        rng = np.random.default_rng(next(self.seed))
                        
                    S = HamiltonianMonteCarloSampler.options(placement_group=pg).remote(self.user,
                                          maxmcmc,
                                          seed        = next(self.seed),
                                          verbose     = self.verbose,
                                          proposal    = proposals['hmc']
                                          )
                    samplers.append(S)
                    
                for i in range(nslice//self.nnest):

                    if self.seed is not None:
                        rng = np.random.default_rng(next(self.seed))
                        
                    S = SliceSampler.options(placement_group=pg).remote(self.user,
                                          maxmcmc,
                                          seed        = next(self.seed),
                                          verbose     = self.verbose,
                                          proposal    = proposals['sli']
                                          )
                    samplers.append(S)
                
                self.pool.append(ActorPool(samplers))
                self.results['run_{}'.format(j)] = {}
                if self.seed is not None:
                    self.results['run_{}'.format(j)]['seed'] = s0.entropy

            self.results['combined'] = {}

    def run(self):
        """
        Run the sampler
        """
        with self.log_file:
            if self.resume:
                signal.signal(signal.SIGTERM, sighandler)
                signal.signal(signal.SIGALRM, sighandler)
                signal.signal(signal.SIGQUIT, sighandler)
                signal.signal(signal.SIGINT, sighandler)
                signal.signal(signal.SIGUSR1, sighandler)
                signal.signal(signal.SIGUSR2, sighandler)
            
            try:
                unfinished = [self.ns_pool[v].nested_sampling_loop.remote(self.pool[v]) for v in  range(self.nnest)]
                while len(unfinished) > 0:
                    finished, unfinished = ray.wait(unfinished)
                    ray.get(finished)

            except CheckPoint:
                self.checkpoint()
                for p in self.ns_pool:
                    ray.kill(p)
                if self.existing_cluster is False:
                    ray.shutdown()
                assert ray.is_initialized() == False
                self.logger.critical("ray correctly shut down. exiting")
                sys.exit(130)

            if self.verbose > 0:
                filename = 'cpnest.h5'
            else:
                filename = None
            
            self.postprocess(filename=filename)

            if self.verbose >= 2:
                self.logger.critical("Checking insertion indeces")
                ray.get([self.ns_pool[v].check_insertion_indices.remote(
                                    rolling=False,
                                    filename='insertion_indices_{}.dat'.format(v))
                                    for v in range(self.nnest)])
 
                self.logger.critical("Saving plots in {0}".format(self.output))
                self.plot(corner = False)

            try:
                for f in self.resume_file:
                    self.logger.info("Removing checkpoint file {}".format(f))
                    os.remove(f)
            except OSError:
                pass
            
            if self.existing_cluster is False:
                ray.shutdown()
                assert ray.is_initialized() == False
                self.logger.critical("ray correctly shut down. exiting")

    def postprocess(self, filename=None):
        """
        post-processes the results of parallel runs
        if filename is not None, then outputs the results in an
        hdf5 file
        """
        import numpy.lib.recfunctions as rfn
        from .nest2pos import draw_posterior_many

        ns = ray.get([self.ns_pool[v].get_nested_samples.remote() for v in range(self.nnest)])
        ps = ray.get([self.ns_pool[v].get_prior_samples.remote() for v in range(self.nnest)])
        info = ray.get([self.ns_pool[v].get_information.remote() for v in range(self.nnest)])

        for i,l in enumerate(ps):

            self.results['run_{}'.format(i)]['prior_samples'] = rfn.stack_arrays([s.asnparray()
                               for s in l] ,usemask=False)

        for i,l in enumerate(ns):
            self.results['run_{}'.format(i)]['nested_samples'] = rfn.stack_arrays([s.asnparray()
                               for s in l] ,usemask=False)

        for i in range(self.nnest):
            p, logZ  = draw_posterior_many([self.results['run_{}'.format(i)]['nested_samples']],
                                           [self.nlive],verbose=self.verbose)

            self.results['run_{}'.format(i)]['posterior_samples'] = p
            self.results['run_{}'.format(i)]['logZ'] = logZ
            self.results['run_{}'.format(i)]['information'] = info[i]
            self.results['run_{}'.format(i)]['logZ_error'] = np.sqrt(info[i]/self.nlive)

        p, logZ  = draw_posterior_many([self.results['run_{}'.format(i)]['nested_samples'] for i in range(self.nnest)],[self.nlive]*self.nnest,verbose=self.verbose,rng=np.random.default_rng(666))

        self.results['combined']['prior_samples'] = rfn.stack_arrays([self.results['run_{}'.format(i)]['prior_samples'] for i in range(self.nnest)],usemask=False)
        self.results['combined']['nested_samples'] = rfn.stack_arrays([self.results['run_{}'.format(i)]['nested_samples'] for i in range(self.nnest)],usemask=False)

        self.results['combined']['nested_samples'].sort(order='logL')
        self.results['combined']['posterior_samples'] = p
        self.results['combined']['logZ'] = logZ
        self.results['combined']['information'] = np.average(info)

        evds = [self.results['run_{}'.format(i)]['logZ'] for i in range(self.nnest)]
        e2   = (np.average(info)/self.nlive+np.var(evds))/self.nnest
        self.results['combined']['logZ_error'] = np.sqrt(e2)

        if filename != None:
            import os
            import h5py
            self.logger.critical("Saving output in {0}".format(self.output))
            h = h5py.File(os.path.join(self.output,filename),'w')
            for k in self.results.keys():
                grp = h.create_group(k)
                for d in self.results[k].keys():
                    grp.create_dataset(d, data = self.results[k][d], dtype=self.results[k][d].dtype)
            h.close()

    @property
    def nested_samples(self):
        return self.results['combined']['nested_samples']

    @nested_samples.setter
    def nested_samples(self, filename=None):
        self.postprocess(self, filename=filename)

    @property
    def posterior_samples(self):
        return self.results['combined']['posterior_samples']

    @posterior_samples.setter
    def posterior_samples(self, filename=None):
        self.postprocess(self, filename=filename)

    @property
    def prior_samples(self):
        return self.results['combined']['prior_samples']

    @prior_samples.setter
    def prior_samples(self, filename=None):
        self.postprocess(self, filename=filename)

    @property
    def logZ(self):
        return self.results['combined']['logZ']

    @logZ.setter
    def logZ(self, filename=None):
        self.postprocess(self, filename=filename)

    @property
    def logZ_error(self):
        return self.results['combined']['logZ_error']

    @logZ_error.setter
    def logZ_error(self, filename=None):
        self.postprocess(self, filename=filename)

    @property
    def information(self):
        return self.results['combined']['information']

    @information.setter
    def information(self, filename=None):
        self.postprocess(self, filename=filename)

    def plot(self, corner = True):
        """
        Make diagnostic plots of the posterior and nested samples
        """
        from . import plot
        for n in self.posterior_samples.dtype.names:
            plot.plot_hist(self.posterior_samples[n].ravel(), name = n,
                           prior_samples = self.prior_samples[n].ravel(),
                           filename = os.path.join(self.output,'posterior_{0}.pdf'.format(n)))

        plot.trace_plot([self.results['run_{}'.format(i)]['nested_samples'] for i in range(self.nnest)],
                        [self.nlive]*self.nnest, self.output)

        if corner:
            import numpy as np
            plotting_posteriors = np.squeeze(self.posterior_samples.view((self.posterior_samples.dtype[0], len(self.posterior_samples.dtype.names))))

            plot.plot_corner(plotting_posteriors,

                             labels=self.prior_samples.dtype.names,
                             filename=os.path.join(self.output,'corner.pdf'))

        lps = ray.get([self.ns_pool[v].get_live_points.remote() for v in range(self.nnest)])

        for i,lp in enumerate(lps):
            plot.plot_indices(lp.get_insertion_indices(), filename=os.path.join(self.output, 'insertion_indices_{}.pdf'.format(i)))

    def checkpoint(self):
        """
        checkpoint the nested samplers
        """
        for i,s in enumerate(self.ns_pool):
            self.logger.critical('Checkpointing nested sampling {}'.format(s))
        ray.get([s.save.remote() for s in self.ns_pool])

    def load_nested_sampler_state(self, resume_file):
        """
        load the nested samplers dictionary state
        """
        self.logger.critical('Loading nested sampling state from {}'.format(resume_file))
        with open(resume_file,"rb") as f:
            obj = dill.load(f)
        return obj
