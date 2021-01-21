#! /usr/bin/env python
# coding: utf-8

import multiprocessing as mp
from ctypes import c_double, c_int
import numpy as np
import os
import sys
import signal
import logging

from multiprocessing.sharedctypes import Value, Array
from multiprocessing import Lock
from multiprocessing.managers import SyncManager

import cProfile

from .utils import LEVELS, LogFile

# module logger takes name according to its path
LOGGER = logging.getLogger('cpnest.cpnest')

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

    poolsize: `int`
        Number of objects in the sampler pool (100)

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
        maximum MCMC points for sampling chains (100)

    nthreads: `int` or `None`
        number of parallel samplers. Default (None) uses mp.cpu_count() to autodetermine

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
                 poolsize     = 100,
                 output       = './',
                 verbose      = 0,
                 seed         = None,
                 maxmcmc      = 100,
                 nthreads     = None,
                 nhamiltonian = 0,
                 nslice       = 0,
                 resume       = False,
                 proposals     = None,
                 n_periodic_checkpoint = None,
                 periodic_checkpoint_interval=None,
                 prior_sampling = False
                 ):
        if nthreads is None:
            self.nthreads = mp.cpu_count()
        else:
            self.nthreads = nthreads

        output = os.path.join(output, '')
        os.makedirs(output, exist_ok=True)

        self.verbose  = verbose
        self.output   = output
        self.logger = logging.getLogger('cpnest.cpnest.CPNest')

        # The LogFile context manager ensures everything within is logged to
        # 'cpnest.log' but the file handler is safely closed once the run is
        # finished.
        self.log_file = LogFile(os.path.join(self.output, 'cpnest.log'),
                                verbose=self.verbose)

        with self.log_file:
            # Everything in this context is logged to `log_file`
            msg = 'Running with {0} parallel threads'.format(self.nthreads)
            self.logger.critical(msg)

            if n_periodic_checkpoint is not None:
                self.logger.critical(
                    "The n_periodic_checkpoint kwarg is deprecated, "
                    "use periodic_checkpoint_interval instead."
                )
            if periodic_checkpoint_interval is None:
                periodic_checkpoint_interval = np.inf

            from .sampler import HamiltonianMonteCarloSampler, MetropolisHastingsSampler, SliceSampler
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
            self.poolsize = poolsize
            self.posterior_samples = None
            self.prior_sampling = prior_sampling
            self.manager = RunManager(
                nthreads=self.nthreads,
                periodic_checkpoint_interval=periodic_checkpoint_interval
            )
            self.manager.start()
            self.user     = usermodel
            self.resume = resume

            if seed is None: self.seed = 1234
            else:
                self.seed=seed

            self.process_pool = []

            # instantiate the nested sampler class
            resume_file = os.path.join(output, "nested_sampler_resume.pkl")
            if not os.path.exists(resume_file) or resume == False:
                self.NS = NestedSampler(self.user,
                            nlive          = nlive,
                            output         = output,
                            verbose        = verbose,
                            seed           = self.seed,
                            prior_sampling = self.prior_sampling,
                            manager        = self.manager)
            else:
                self.NS = NestedSampler.resume(resume_file, self.manager,
                                               self.user)

            nmhs = self.nthreads-nhamiltonian-nslice
            # instantiate the sampler class
            for i in range(nmhs):
                resume_file = os.path.join(output,
                                           "sampler_{0:d}.pkl".format(i))
                if not os.path.exists(resume_file) or resume == False:
                    sampler = MetropolisHastingsSampler(self.user,
                                    maxmcmc,
                                    verbose     = verbose,
                                    output      = output,
                                    poolsize    = poolsize,
                                    seed        = self.seed+i,
                                    proposal    = proposals['mhs'](),
                                    resume_file = resume_file,
                                    sample_prior = prior_sampling,
                                    manager     = self.manager
                                    )
                else:
                    sampler = MetropolisHastingsSampler.resume(resume_file,
                                                            self.manager,
                                                            self.user)

                p = mp.Process(target=sampler.produce_sample)
                self.process_pool.append(p)

            for i in range(nhamiltonian):
                resume_file = os.path.join(output,
                                           "sampler_{0:d}.pkl".format(i))
                if not os.path.exists(resume_file) or resume == False:
                    sampler = HamiltonianMonteCarloSampler(
                        self.user,
                        maxmcmc,
                        verbose     = verbose,
                        output      = output,
                        poolsize    = poolsize,
                        seed        = self.seed+nmhs+i,
                        proposal    = proposals['hmc'](model=self.user),
                        resume_file = resume_file,
                        sample_prior = prior_sampling,
                        manager     = self.manager
                    )
                else:
                    sampler = HamiltonianMonteCarloSampler.resume(resume_file,
                                                                self.manager,
                                                                self.user)
                p = mp.Process(target=sampler.produce_sample)
                self.process_pool.append(p)

            for i in range(nslice):
                resume_file = os.path.join(output, "sampler_{0:d}.pkl".format(i))
                if not os.path.exists(resume_file) or resume == False:
                    sampler = SliceSampler(self.user,
                                    maxmcmc,
                                    verbose     = verbose,
                                    output      = output,
                                    poolsize    = poolsize,
                                    seed        = self.seed+nmhs+nhamiltonian+i,
                                    proposal    = proposals['sli'](),
                                    resume_file = resume_file,
                                    manager     = self.manager
                                    )
                else:
                    sampler = SliceSampler.resume(resume_file,
                                                  self.manager,
                                                  self.user)
                p = mp.Process(target=sampler.produce_sample)
                self.process_pool.append(p)

    def run(self):
        """
        Run the sampler
        """

        with self.log_file:
            # Everything in this context is logged to `log_file`

            if self.resume:
                signal.signal(signal.SIGTERM, sighandler)
                signal.signal(signal.SIGALRM, sighandler)
                signal.signal(signal.SIGQUIT, sighandler)
                signal.signal(signal.SIGINT, sighandler)
                signal.signal(signal.SIGUSR1, sighandler)
                signal.signal(signal.SIGUSR2, sighandler)

            #self.p_ns.start()
            for each in self.process_pool:
                each.start()
            try:
                self.NS.nested_sampling_loop()
                for each in self.process_pool:
                    each.join()
            except CheckPoint:
                self.checkpoint()
                sys.exit(130)

            if self.verbose >= 2:
                self.NS.check_insertion_indices(rolling=False,
                                                filename='insertion_indices.dat')
                self.logger.critical(
                    "Saving nested samples in {0}".format(self.output)
                )
                self.nested_samples = self.get_nested_samples()
                self.logger.critical(
                    "Saving posterior samples in {0}".format(self.output)
                )
                self.posterior_samples = self.get_posterior_samples()
            else:
                self.NS.check_insertion_indices(rolling=False,
                                                filename=None)
                self.nested_samples = self.get_nested_samples(filename=None)
                self.posterior_samples = self.get_posterior_samples(
                    filename=None
                )

            if self.verbose>=3 or self.NS.prior_sampling:
                self.prior_samples = self.get_prior_samples(filename=None)
            if self.verbose>=3 and not self.NS.prior_sampling:
                self.mcmc_samples = self.get_mcmc_samples(filename=None)
            if self.verbose>=2:
                self.plot(corner = False)

            #TODO: Clean up the resume pickles

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
        import numpy.lib.recfunctions as rfn
        self.nested_samples = rfn.stack_arrays(
                [s.asnparray()
                    for s in self.NS.nested_samples]
                ,usemask=False)
        if filename:
            np.savetxt(os.path.join(
                self.NS.output_folder, filename),
                self.nested_samples.ravel(),
                header=' '.join(self.nested_samples.dtype.names),
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
        posterior_samples  = draw_posterior_many([nested_samples],[self.nlive],verbose=self.verbose)
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
            for file in os.listdir(self.NS.output_folder):
                if 'prior_samples' in file:
                    prior_samples.append(np.genfromtxt(os.path.join(self.NS.output_folder,file), names = True))
                    os.system('rm {0}'.format(os.path.join(self.NS.output_folder,file)))
                elif 'mcmc_chain' in file:
                    mcmc_samples.append(resample_mcmc_chain(np.genfromtxt(os.path.join(self.NS.output_folder,file), names = True)))
                    os.system('rm {0}'.format(os.path.join(self.NS.output_folder,file)))

            # first deal with the prior samples
            if len(prior_samples)>0:
                self.prior_samples = stack_arrays([p for p in prior_samples])
                if filename:
                    np.savetxt(os.path.join(
                           self.NS.output_folder,'prior.dat'),
                           self.prior_samples.ravel(),
                           header=' '.join(self.prior_samples.dtype.names),
                           newline='\n',delimiter=' ')
            # now stack all the mcmc chains
            if len(mcmc_samples)>0:
                self.mcmc_samples = stack_arrays([p for p in mcmc_samples])
                if filename:
                    np.savetxt(os.path.join(
                           self.NS.output_folder,'mcmc.dat'),
                           self.mcmc_samples.ravel(),
                           header=' '.join(self.mcmc_samples.dtype.names),
                           newline='\n',delimiter=' ')

        # TODO: Replace with something to output samples in whatever format
        if filename:
            np.savetxt(os.path.join(
                self.NS.output_folder, filename),
                posterior_samples.ravel(),
                header=' '.join(posterior_samples.dtype.names),
                newline='\n',delimiter=' ')
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
        for file in os.listdir(self.NS.output_folder):
            if 'prior_samples' in file:
                prior_samples.append(np.genfromtxt(os.path.join(self.NS.output_folder,file), names = True))
                os.system('rm {0}'.format(os.path.join(self.NS.output_folder,file)))

        # if we sampled the prior, the nested samples are samples from the prior
        if self.NS.prior_sampling:
            prior_samples.append(self.get_nested_samples())

        if not prior_samples:
            self.logger.critical('ERROR, no prior samples found!')
            return None

        prior_samples = stack_arrays([p for p in prior_samples])
        if filename:
            np.savetxt(os.path.join(
                       self.NS.output_folder, filename),
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
        for file in os.listdir(self.NS.output_folder):
            if 'mcmc_chain' in file:
                mcmc_samples.append(resample_mcmc_chain(np.genfromtxt(os.path.join(self.NS.output_folder,file), names = True)))
                os.system('rm {0}'.format(os.path.join(self.NS.output_folder,file)))

        if not mcmc_samples:
            self.logger.critical('ERROR, no MCMC samples found!')
            return None

        # now stack all the mcmc chains
        mcmc_samples = stack_arrays([p for p in mcmc_samples])
        if filename:
            np.savetxt(os.path.join(
                       self.NS.output_folder, filename),
                       self.mcmc_samples.ravel(),
                       header=' '.join(self.mcmc_samples.dtype.names),
                       newline='\n',delimiter=' ')
        return mcmc_samples

    def plot(self, corner = True):
        """
        Make diagnostic plots of the posterior and nested samples
        """
        pos = self.posterior_samples
        if self.verbose>=3 and self.NS.prior_sampling is False:
            pri = self.prior_samples
            mc  = self.mcmc_samples
        elif self.verbose>=3 or self.NS.prior_sampling is True:
            pri = self.prior_samples
            mc  = None
        else:
            pri = None
            mc  = None
        from . import plot
        if self.NS.prior_sampling is False:
            for n in pos.dtype.names:
                plot.plot_hist(pos[n].ravel(), name = n,
                               prior_samples = self.prior_samples[n].ravel() if pri is not None else None,
                               mcmc_samples = self.mcmc_samples[n].ravel() if mc is not None else None,
                               filename = os.path.join(self.output,'posterior_{0}.pdf'.format(n)))
        for n in self.nested_samples.dtype.names:
            plot.plot_chain(self.nested_samples[n],name=n,filename=os.path.join(self.output,'nschain_{0}.pdf'.format(n)))
        if self.NS.prior_sampling is False:
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
            plot.plot_indices(self.NS.insertion_indices, filename=os.path.join(self.output, 'insertion_indices.pdf'))

    def worker_sampler(self, producer_pipe, logLmin):
        cProfile.runctx('self.sampler.produce_sample(producer_pipe, logLmin)', globals(), locals(), 'prof_sampler.prof')

    def worker_ns(self):
        cProfile.runctx('self.NS.nested_sampling_loop(self.consumer_pipes)', globals(), locals(), 'prof_nested_sampling.prof')

    def profile(self):
        for i in range(0,self.NUMBER_OF_PRODUCER_PROCESSES):
            p = mp.Process(target=self.worker_sampler, args=(self.queues[i%len(self.queues)], self.NS.logLmin ))
            self.process_pool.append(p)
        for i in range(0,self.NUMBER_OF_CONSUMER_PROCESSES):
            p = mp.Process(target=self.worker_ns, args=(self.queues, self.port, self.authkey))
            self.process_pool.append(p)
        for each in self.process_pool:
            each.start()

    def checkpoint(self):
        self.manager.checkpoint_flag=1


class RunManager(SyncManager):
    def __init__(self, nthreads=None, **kwargs):
        self.periodic_checkpoint_interval = kwargs.pop(
            "periodic_checkpoint_interval", np.inf
        )
        super(RunManager,self).__init__(**kwargs)
        self.nconnected=mp.Value(c_int,0)
        self.producer_pipes = list()
        self.consumer_pipes = list()
        for i in range(nthreads):
            consumer, producer = mp.Pipe(duplex=True)
            self.producer_pipes.append(producer)
            self.consumer_pipes.append(consumer)
        self.logLmin = None
        self.logLmax = None
        self.nthreads=nthreads

    def start(self):
        super(RunManager, self).start()
        self.logLmin = mp.Value(c_double,-np.inf)
        self.logLmax = mp.Value(c_double,-np.inf)
        self.checkpoint_flag=mp.Value(c_int,0)

    def connect_producer(self):
        """
        Returns the producer's end of the pipe
        """
        with self.nconnected.get_lock():
            n = self.nconnected.value
            pipe = self.producer_pipes[n]
            self.nconnected.value+=1
        return pipe, n
