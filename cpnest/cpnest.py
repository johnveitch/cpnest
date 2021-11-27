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
        maximum MCMC points for MHS sampling chains (100)

    maxslice: `int`
        maximum number of slices points for Slice sampling chains (100)

    maxleaps: `int`
        maximum number of leaps points for HMC sampling chains (100)

    nthreads: `int` or `None`
        number of parallel samplers. Default (None) uses psutil.cpu_count() to autodetermine

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

    """
    def __init__(self,
                 usermodel,
                 nlive        = 100,
                 output       = './',
                 verbose      = 0,
                 seed         = None,
                 maxmcmc      = 5000,
                 nnest        = 1,
                 nensemble    = 0,
                 nhamiltonian = 0,
                 nslice       = 0,
                 resume       = False,
                 proposals     = None,
                 n_periodic_checkpoint = None,
                 periodic_checkpoint_interval=None,
                 prior_sampling = False,
                 object_store_memory=2*10**9,
                 poolsize=None,
                 nthreads=None
                 ):


        self.logger    = logging.getLogger('cpnest.cpnest.CPNest')
        self.nsamplers = nensemble+nhamiltonian+nslice
        self.nnest     = nnest
        if nthreads is not None and self.nsamplers == 0:
            nensemble = nthreads
            self.nsamplers = nensemble
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

        # The LogFile context manager ensures everything within is logged to
        # 'cpnest.log' but the file handler is safely closed once the run is
        # finished.
        self.log_file = LogFile(os.path.join(output, 'cpnest.log'),
                                verbose=verbose)
        with self.log_file:
            if poolsize is not None:
                self.logger.warning('poolsize is a deprecated option and will \
                                    be removed in a future version.')
            if nthreads is not None:
                self.logger.warning('nthreads is a deprecated option and will\
                                    be removed in a future verison.')
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
            self.nested_samples = None
            self.posterior_samples = None
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
                                          nlive       = nlive,
                                          proposal    = proposals['mhs']()
                                          )
                    samplers.append(s)

                for i in range(nhamiltonian//self.nnest):
                    s = HamiltonianMonteCarloSampler.options(placement_group=pg).remote(self.user,
                                          maxmcmc,
                                          verbose     = verbose,
                                          nlive       = nlive,
                                          proposal    = proposals['hmc'](model=self.user)
                                          )
                    samplers.append(s)

                for i in range(nslice//self.nnest):
                    s = SliceSampler.options(placement_group=pg).remote(self.user,
                                          maxmcmc,
                                          verbose     = verbose,
                                          nlive       = nlive,
                                          proposal    = proposals['sli']()
                                          )
                    samplers.append(s)

                self.pool.append(ActorPool(samplers))

                self.resume_file = os.path.join(output, "nested_sampler_resume_{}.pkl".format(j))
                if not os.path.exists(self.resume_file) or resume == False:
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
                    self.ns_pool.append(ray.remote(NestedSampler).resume(self.resume_file, self.user, self.pool[i]))

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
                sys.exit(130)

            if self.verbose >= 2:
                for s in self.NS.map_unordered(lambda a, v: a.check_insertion_indices.remote(rolling=False,
                                                     filename='insertion_indices_{}.dat'.format(v)), range(self.nnest)):
                    pass

                self.logger.critical(
                    "Saving nested samples in {0}".format(self.output)
                )
                self.nested_samples = self.get_nested_samples()
                self.logger.critical("Saving posterior samples in {0}".format(self.output))
                self.posterior_samples = self.get_posterior_samples()
            else:
                for s in self.NS.map_unordered(lambda a, v: a.check_insertion_indices.remote(rolling=False,
                                                     filename=None), range(self.nnest)):
                    pass
                self.nested_samples = self.get_nested_samples(filename=None)
                self.posterior_samples = self.get_posterior_samples(
                    filename=None
                )

            if self.verbose>=3 or self.prior_sampling:
                self.prior_samples = self.get_prior_samples(filename=None)
            if self.verbose>=3 and not self.prior_sampling:
                self.mcmc_samples = self.get_mcmc_samples(filename=None)
            if self.verbose>=2:
                self.plot(corner = False)

            #TODO: Clean up the resume pickles
            try:
                os.remove(self.resume_file)
            except OSError:
                pass

            ray.shutdown()
            assert ray.is_initialized() == False

    def get_nested_samples(self, filename='nested_samples.dat'):
        """
        returns nested sampling chain
        Parameters
        ----------
        filename : string
                   If given, file to save nested samples to

        Returns
        -------
        pos : :obj:`numpy.ndarray`
        """

        self.nested_samples = None
        import numpy.lib.recfunctions as rfn
        if self.nested_samples is None:
            ns = list(self.NS.map_unordered(lambda a, v: a.get_nested_samples.remote(), range(self.nnest)))

            self.nested_samples = []

            for l in ns:
                self.nested_samples.append(rfn.stack_arrays([s.asnparray()
                                   for s in l] ,usemask=False))


        if filename:
            ns_samps = rfn.stack_arrays(self.nested_samples,usemask=False)
            ns_samps.sort(order='logL')
            np.savetxt(os.path.join(
                self.output, filename),
                ns_samps.ravel(),
                header=' '.join(ns_samps.dtype.names),
                newline='\n',delimiter=' ')
        return self.nested_samples

    def get_posterior_samples(self, filename='posterior.dat'):
        """
        Returns posterior samples

        Parameters
        ----------
        filename : string
                   If given, file to save posterior samples to

        Returns
        -------
        pos : :obj:`numpy.ndarray`
        """
        import numpy as np
        import os
        from .nest2pos import draw_posterior_many
        nested_samples     = self.get_nested_samples()
        posterior_samples, self.logZ  = draw_posterior_many(nested_samples,[self.nlive]*self.nnest,verbose=self.verbose)
        posterior_samples  = np.array(posterior_samples)
        self.prior_samples = {n:None for n in self.user.names}
        self.mcmc_samples  = {n:None for n in self.user.names}
        # if we run with full verbose, read in and output
        # the mcmc thinned posterior samples
        if self.verbose >= 3:
            from .nest2pos import resample_mcmc_chain
            from numpy.lib.recfunctions import stack_arrays

            prior_samples = []
            mcmc_samples  = []
            for file in os.listdir(self.output):
                if 'prior_samples' in file:
                    prior_samples.append(np.genfromtxt(os.path.join(self.output,file), names = True))
                    os.system('rm {0}'.format(os.path.join(self.output,file)))
                elif 'mcmc_chain' in file:
                    mcmc_samples.append(resample_mcmc_chain(np.genfromtxt(os.path.join(self.output,file), names = True)))
                    os.system('rm {0}'.format(os.path.join(self.output,file)))

            # first deal with the prior samples
            if len(prior_samples)>0:
                self.prior_samples = stack_arrays([p for p in prior_samples])
                if filename:
                    np.savetxt(os.path.join(
                           self.output,'prior.dat'),
                           self.prior_samples.ravel(),
                           header=' '.join(self.prior_samples.dtype.names),
                           newline='\n',delimiter=' ')
            # now stack all the mcmc chains
            if len(mcmc_samples)>0:
                self.mcmc_samples = stack_arrays([p for p in mcmc_samples])
                if filename:
                    np.savetxt(os.path.join(
                           self.output,'mcmc.dat'),
                           self.mcmc_samples.ravel(),
                           header=' '.join(self.mcmc_samples.dtype.names),
                           newline='\n',delimiter=' ')

        # TODO: Replace with something to output samples in whatever format
        if filename:
            np.savetxt(os.path.join(
                self.output, filename),
                posterior_samples.ravel(),
                header=' '.join(posterior_samples.dtype.names),
                newline='\n',delimiter=' ')
            np.savetxt(os.path.join(
                self.output, filename+"_evidence"),np.atleast_1d(self.logZ))


        return posterior_samples

    def get_prior_samples(self, filename='prior.dat'):
        """
        Returns prior samples

        Parameters
        ----------
        filename : string
                   If given, file to save posterior samples to

        Returns
        -------
        pos : :obj:`numpy.ndarray`
        """
        import numpy as np
        import os

        from numpy.lib.recfunctions import stack_arrays

        # read in the samples from the prior coming from each sampler
        prior_samples = []
        for file in os.listdir(self.output):
            if 'prior_samples' in file:
                prior_samples.append(np.genfromtxt(os.path.join(self.output,file), names = True))
                os.system('rm {0}'.format(os.path.join(self.output,file)))

        # if we sampled the prior, the nested samples are samples from the prior
        if self.prior_sampling:
            prior_samples.append(self.get_nested_samples())

        if not prior_samples:
            self.logger.critical('ERROR, no prior samples found!')
            return None

        prior_samples = stack_arrays([p for p in prior_samples])
        if filename:
            np.savetxt(os.path.join(
                       self.output, filename),
                       self.prior_samples.ravel(),
                       header=' '.join(self.prior_samples.dtype.names),
                       newline='\n',delimiter=' ')

        return prior_samples

    def get_mcmc_samples(self, filename='mcmc.dat'):
        """
        Returns resampled mcmc samples

        Parameters
        ----------
        filename : string
                   If given, file to save posterior samples to

        Returns
        -------
        pos : :obj:`numpy.ndarray`
        """
        import numpy as np
        import os
        from .nest2pos import resample_mcmc_chain
        from numpy.lib.recfunctions import stack_arrays

        mcmc_samples  = []
        for file in os.listdir(self.output):
            if 'mcmc_chain' in file:
                mcmc_samples.append(resample_mcmc_chain(np.genfromtxt(os.path.join(self.output,file), names = True)))
                os.system('rm {0}'.format(os.path.join(self.output,file)))

        if not mcmc_samples:
            self.logger.critical('ERROR, no MCMC samples found!')
            return None

        # now stack all the mcmc chains
        mcmc_samples = stack_arrays([p for p in mcmc_samples])
        if filename:
            np.savetxt(os.path.join(
                       self.output, filename),
                       self.mcmc_samples.ravel(),
                       header=' '.join(self.mcmc_samples.dtype.names),
                       newline='\n',delimiter=' ')
        return mcmc_samples

    def plot(self, corner = True):
        """
        Make diagnostic plots of the posterior and nested samples
        """
        pos = self.posterior_samples
        if self.verbose>=3 and self.prior_sampling is False:
            pri = self.prior_samples
            mc  = self.mcmc_samples
        elif self.verbose>=3 or self.prior_sampling is True:
            pri = self.prior_samples
            mc  = None
        else:
            pri = None
            mc  = None
        from . import plot
        if self.prior_sampling is False:
            for n in pos.dtype.names:
                plot.plot_hist(pos[n].ravel(), name = n,
                               prior_samples = self.prior_samples[n].ravel() if pri is not None else None,
                               mcmc_samples = self.mcmc_samples[n].ravel() if mc is not None else None,
                               filename = os.path.join(self.output,'posterior_{0}.pdf'.format(n)))

        plot.trace_plot(self.nested_samples,[self.nlive]*self.nnest,self.output)

        if self.prior_sampling is False:
            import numpy as np
            plotting_posteriors = np.squeeze(pos.view((pos.dtype[0], len(pos.dtype.names))))
            if pri is not None:
                plotting_priors = np.squeeze(pri.view((pri.dtype[0], len(pri.dtype.names))))
            else:
                plotting_priors = None

            if mc is not None:
                plotting_mcmc   = np.squeeze(mc.view((mc.dtype[0], len(mc.dtype.names))))
            else:
                plotting_mcmc   = None

            if corner:
                plot.plot_corner(plotting_posteriors,
                                 ps=plotting_priors,
                                 ms=plotting_mcmc,
                                 labels=pos.dtype.names,
                                 filename=os.path.join(self.output,'corner.pdf'))
            lps = list(self.NS.map(lambda a, v: a.get_live_points.remote(), range(self.nnest)))
            for i,lp in enumerate(lps):
                plot.plot_indices(lp.get_insertion_indices(), filename=os.path.join(self.output, 'insertion_indices_{}.pdf'.format(i)))

    def checkpoint(self):
        self.NS.checkpoint()
