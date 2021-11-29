#! /usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
import signal
import logging

import cProfile

from .utils import LEVELS, LogFile, auto_garbage_collect

# module logger takes name according to its path
LOGGER = logging.getLogger('cpnest.cpnest')
import ray
from ray.util import ActorPool

class CheckPoint(Exception):
    pass

def sighandler(signal, frame):
    # print("Handling signal {}".format(signal))
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

    maxslice: `int`
        maximum number of slices points for Slice sampling chains (100)

    maxleaps: `int`
        maximum number of leaps points for HMC sampling chains (1000)

    nnest: `int`
        number of parallel nested samplers.
        Default: 1

    nensemble: `int`
        number of sampler threads using an ensemble samplers. Default: 1
    
    nhamiltonian: `int`
        number of sampler threads using an hamiltonian samplers. Default: 0

    nslice: `int`
            number of sampler threads using an ensemble slice samplers. Default: 0

    resume: `boolean`
        determines whether cpnest will resume a run or run from scratch. Default: False.

    proposal: `dict`
        dictionary of lists with custom jump proposals.
        key 'mhs' for the Metropolis-Hastings sampler,
        'hmc' for the Hamiltonian Monte-Carlo sampler,
        'sli' for the slice sampler.
        'hmc' for the Hamiltonian Monte-Carlo sampler. Default: None

    prior_sampling: `boolean`
        generates samples from the prior

    n_periodic_checkpoint: `int`
        **deprecated**
        checkpoint the sampler every n_periodic_checkpoint iterations
        Default: None (disabled)

    periodic_checkpoint_interval: `float`
        checkpoing the sampler every periodic_checkpoint_interval seconds
        Default: None (disabled)

    prior_sampling: boolean
        produce nlive samples from the prior.
        Default: False
    
    object_store_memory: `int`
        amount of memory reserved for ray object store
        Default: 2GB

    """
    def __init__(self,
                 usermodel,
                 nlive        = 100,
                 output       = './',
                 verbose      = 0,
                 seed         = None,
                 maxmcmc      = 5000,
                 maxslice     = 100,
                 maxleaps     = 1000,
                 nnest        = 1,
                 nensemble    = 1,
                 nhamiltonian = 0,
                 nslice       = 0,
                 resume       = False,
                 proposals     = None,
                 n_periodic_checkpoint = None,
                 periodic_checkpoint_interval=None,
                 prior_sampling = False,
                 object_store_memory=2*10**9
                 ):
        
        self.logger    = logging.getLogger('cpnest.cpnest.CPNest')
        self.nsamplers = nensemble+nhamiltonian+nslice
        self.nnest     = nnest
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
        
        ray.init(num_cpus=self.nthreads,
                 ignore_reinit_error=True,
                 object_store_memory=object_store_memory)
        
        assert ray.is_initialized() == True
        output = os.path.join(output, '')
        os.makedirs(output, exist_ok=True)

        # Import placement group APIs.
        from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group

        # set up the resume files
        self.resume_file = []
        # The LogFile context manager ensures everything within is logged to
        # 'cpnest.log' but the file handler is safely closed once the run is
        # finished.
        self.log_file = LogFile(os.path.join(output, 'cpnest.log'),
                                verbose=verbose)
        with self.log_file:
            self.logger.critical('Running with {0} parallel threads'.format(self.nthreads))
            self.logger.critical('Nested samplers: {0}'.format(nnest))
            self.logger.critical('Ensemble samplers: {0}'.format(nensemble))
            self.logger.critical('Slice samplers: {0}'.format(nslice))
            self.logger.critical('Hamiltonian samplers: {0}'.format(nhamiltonian))
            self.logger.critical('ray object store size: {0} GB'.format(object_store_memory/1e9))
            
            if n_periodic_checkpoint is not None:
                self.logger.critical(
                    "The n_periodic_checkpoint kwarg is deprecated, "
                    "use periodic_checkpoint_interval instead."
                )
            if periodic_checkpoint_interval is None:
                periodic_checkpoint_interval = np.inf

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
            self.verbose  = verbose
            self.output   = output
            self.results  = {}
            self.prior_sampling = prior_sampling
            self.user     = usermodel
            self.resume = resume

            if seed is None: self.seed=1234
            else:
                self.seed=seed
            
            for j in range(self.nnest):
            
                pg = placement_group([{"CPU": 1+self.nsamplers//self.nnest}],strategy="STRICT_PACK")
                ray.get(pg.ready())
                
                samplers = []
                
                # instantiate the sampler class
                for i in range(nensemble//self.nnest):
                    s = MetropolisHastingsSampler.options(placement_group=pg).remote(self.user,
                                          maxmcmc,
                                          verbose     = verbose,
                                          proposal    = proposals['mhs']()
                                          )
                    samplers.append(s)

                for i in range(nhamiltonian//self.nnest):
                    s = HamiltonianMonteCarloSampler.options(placement_group=pg).remote(self.user,
                                          maxmcmc,
                                          verbose     = verbose,
                                          proposal    = proposals['hmc'](model=self.user)
                                          )
                    samplers.append(s)

                for i in range(nslice//self.nnest):
                    s = SliceSampler.options(placement_group=pg).remote(self.user,
                                          maxmcmc,
                                          verbose     = verbose,
                                          proposal    = proposals['sli']()
                                          )
                    samplers.append(s)
                
                self.pool.append(ActorPool(samplers))

                self.resume_file.append(os.path.join(output, "nested_sampler_resume_{}.pkl".format(j)))
                if not os.path.exists(self.resume_file[j]) or resume == False:
                    self.ns_pool.append(ray.remote(NestedSampler).options(placement_group=pg).remote(self.user,
                                nthreads       = self.nsamplers,
                                nlive          = nlive,
                                output         = output,
                                verbose        = verbose,
                                seed           = self.seed+j,
                                prior_sampling = self.prior_sampling,
                                periodic_checkpoint_interval = periodic_checkpoint_interval,
                                position = j))
                else:
                    self.ns_pool.append(ray.remote(NestedSampler).resume(self.resume_file[j], self.user, self.pool[i]))
                
                self.results['run_{}'.format(j)] = {}
            
            self.results['combined'] = {}
            self.NS = ActorPool(self.ns_pool)
            
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
                for s in self.NS.map_unordered(lambda a, v: a.nested_sampling_loop.remote(self.pool[v]), range(self.nnest)):
                    pass
            
            except CheckPoint:
                self.checkpoint()
                for p in self.NS+self.samplers:
                    p.shutdown()
                ray.shutdown()
                assert ray.is_initialized() == False
                self.logger.critical("ray correctly shut down. exiting")
                sys.exit(130)

            self.postprocess(filename='cpnest.h5')
            
            if self.verbose >= 2:
                self.logger.critical("Checking insertion indeces")
                for s in self.NS.map_unordered(lambda a, v: a.check_insertion_indices.remote(rolling=False,
                                                     filename='insertion_indices_{}.dat'.format(v)), range(self.nnest)):
                    pass

                self.logger.critical("Saving plots in {0}".format(self.output))
                self.plot(corner = False)
                
            
            #TODO: Clean up the resume pickles
            try:
                for f in self.resume_file:
                    os.remove(f)
            except OSError:
                pass
                        
            ray.shutdown()
            assert ray.is_initialized() == False
            self.logger.critical("ray correctly shut down. exiting")
            
    def postprocess(self, filename='cpnest.h5'):
        """
        post-processes the results of parallel runs
        if filenmae is not None, then outputs the results in an
        hdf5 file
        """
        import numpy.lib.recfunctions as rfn
        from .nest2pos import draw_posterior_many
        
        ns = list(self.NS.map_unordered(lambda a, v: a.get_nested_samples.remote(), range(self.nnest)))
        ps = list(self.NS.map_unordered(lambda a, v: a.get_prior_samples.remote(), range(self.nnest)))
        info = list(self.NS.map_unordered(lambda a, v: a.get_information.remote(), range(self.nnest)))

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
            
        p, logZ  = draw_posterior_many([self.results['run_{}'.format(i)]['nested_samples'] for i in range(self.nnest)],[self.nlive]*self.nnest,verbose=self.verbose)
        
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
        
        lps = list(self.NS.map(lambda a, v: a.get_live_points.remote(), range(self.nnest)))
        for i,lp in enumerate(lps):
            plot.plot_indices(lp.get_insertion_indices(), filename=os.path.join(self.output, 'insertion_indices_{}.pdf'.format(i)))

    def checkpoint(self):
        """
        send the checkpoint message to the nested samplers
        """
        for s in self.NS.map_unordered(lambda a, v: a.checkpoint.remote(), range(self.nnest)):
            pass
