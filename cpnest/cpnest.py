#! /usr/bin/env python
# coding: utf-8

import multiprocessing as mp
from ctypes import c_double, c_int
import numpy as np
import os
import sys
import signal

from multiprocessing.sharedctypes import Value
from multiprocessing import Lock
from multiprocessing.managers import SyncManager

import cProfile
import time
import os

class CheckPoint(Exception):
    pass

def sighandler(signal, frame):
    raise CheckPoint

class CPNest(object):
    """
    Class to control CPNest sampler
    cp = CPNest(usermodel,nlive=100,output='./',verbose=0,seed=None,maxmcmc=100,nthreads=None,balanced_sampling = True)
    
    Input variables:
    usermodel : an object inheriting cpnest.model.Model that defines the user's problem
    nlive : Number of live points (100)
    poolsize: Number of objects in the sampler pool (100)
    output : output directory (./)
    verbose: Verbosity, 0=silent, 1=progress, 2=diagnostic, 3=detailed diagnostic
    seed: random seed (default: 1234)
    maxmcmc: maximum MCMC points for sampling chains (100)
    nthreads: number of parallel samplers. Default (None) uses mp.cpu_count() to autodetermine
    nhamiltomnian: number of sampler threads using an hamiltonian samplers. Default: 0
    resume: determines whether cpnest will resume a run or run from scratch. Default: False.
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
                 resume       = False):
        if nthreads is None:
            self.nthreads = mp.cpu_count()
        else:
            self.nthreads = nthreads
        print('Running with {0} parallel threads'.format(self.nthreads))
        from .sampler import HamiltonianMonteCarloSampler, MetropolisHastingsSampler
        from .NestedSampling import NestedSampler
        from .proposal import DefaultProposalCycle, HamiltonianProposalCycle
        self.user     = usermodel
        self.nlive    = nlive
        self.verbose  = verbose
        self.output   = output
        self.poolsize = poolsize
        self.posterior_samples = None
        self.manager = RunManager(nthreads=self.nthreads)
        self.manager.start()
        self.resume = resume

        if seed is None: self.seed=1234
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
                        prior_sampling = False,
                        manager        = self.manager)
        else:
            self.NS = NestedSampler.resume(resume_file, self.manager, self.user)

        # instantiate the sampler class
        for i in range(self.nthreads-nhamiltonian):
            resume_file = os.path.join(output, "sampler_{0:d}.pkl".format(i))
            if not os.path.exists(resume_file) or resume == False:
                sampler = MetropolisHastingsSampler(self.user,
                                  maxmcmc,
                                  verbose     = verbose,
                                  output      = output,
                                  poolsize    = poolsize,
                                  seed        = self.seed+i,
                                  proposal    = DefaultProposalCycle(),
                                  resume_file = resume_file,
                                  manager     = self.manager
                                  )
            else:
                sampler = MetropolisHastingsSampler.resume(resume_file,
                                                           self.manager,
                                                           self.user)

            p = mp.Process(target=sampler.produce_sample)
            self.process_pool.append(p)
        
        for i in range(self.nthreads-nhamiltonian,self.nthreads):
            resume_file = os.path.join(output, "sampler_{0:d}.pkl".format(i))
            if not os.path.exists(resume_file) or resume == False:
                sampler = HamiltonianMonteCarloSampler(self.user,
                                  maxmcmc,
                                  verbose     = verbose,
                                  output      = output,
                                  poolsize    = poolsize,
                                  seed        = self.seed+i,
                                  proposal    = HamiltonianProposalCycle(model=self.user),
                                  resume_file = resume_file,
                                  manager     = self.manager
                                  )
            else:
                sampler = HamiltonianMonteCarloSampler.resume(resume_file,
                                                              self.manager,
                                                              self.user)
            p = mp.Process(target=sampler.produce_sample)
            self.process_pool.append(p)

    def run(self):
        """
        Run the sampler
        """
        if self.resume:
            signal.signal(signal.SIGTERM, sighandler)
            signal.signal(signal.SIGQUIT, sighandler)
            signal.signal(signal.SIGINT, sighandler)
        
        #self.p_ns.start()
        for each in self.process_pool:
            each.start()
        try:
            self.NS.nested_sampling_loop()
            for each in self.process_pool:
                each.join()
        except CheckPoint:
            self.checkpoint()
            sys.exit()

        self.posterior_samples = self.get_posterior_samples(filename=None)
        if self.verbose>1: self.plot()
    
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
                self.NS.output_folder,'nested_samples.dat'),
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
        nested_samples = self.get_nested_samples()
        posterior_samples = draw_posterior_many([nested_samples],[self.nlive],verbose=self.verbose)
        posterior_samples = np.array(posterior_samples)
        # TODO: Replace with something to output samples in whatever format
        if filename:
            np.savetxt(os.path.join(
                self.NS.output_folder,'posterior.dat'),
                self.posterior_samples.ravel(),
                header=' '.join(posterior_samples.dtype.names),
                newline='\n',delimiter=' ')
        return posterior_samples

    def plot(self):
        """
        Make diagnostic plots of the posterior and nested samples
        """
        pos = self.posterior_samples
        from . import plot
        for n in pos.dtype.names:
            plot.plot_hist(pos[n].ravel(),name=n,filename=os.path.join(self.output,'posterior_{0}.png'.format(n)))
        for n in self.nested_samples.dtype.names:
            plot.plot_chain(self.nested_samples[n],name=n,filename=os.path.join(self.output,'nschain_{0}.png'.format(n)))
        import numpy as np
        plotting_posteriors = np.squeeze(pos.view((pos.dtype[0], len(pos.dtype.names))))
        plot.plot_corner(plotting_posteriors,labels=pos.dtype.names,filename=os.path.join(self.output,'corner.png'))

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
        super(RunManager,self).__init__(**kwargs)
        self.nconnected=0
        self.producer_pipes = list()
        self.consumer_pipes = list()
        for i in range(nthreads):
            consumer, producer = mp.Pipe(duplex=True)
            self.producer_pipes.append(producer)
            self.consumer_pipes.append(consumer)
        self.logLmin=None
        self.nthreads=nthreads

    def start(self):
        super(RunManager, self).start()
        self.logLmin = mp.Value(c_double,-np.inf)
        self.checkpoint_flag=mp.Value(c_int,0)

    def connect_producer(self):
        """
        Returns the producer's end of the pipe
        """
        self.nconnected+=1
        return self.producer_pipes[self.nconnected-1]
