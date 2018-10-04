#! /usr/bin/env python
# coding: utf-8

import multiprocessing as mp
import cProfile
import time
import os

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
    """
    def __init__(self,
                 usermodel,
                 nlive      = 100,
                 poolsize   = 100,
                 output     = './',
                 verbose    = 0,
                 seed       = None,
                 maxmcmc    = 100,
                 nthreads   = None,
                 nhamiltonian = 0,
                 resume_file  = None):
        if nthreads is None:
            self.nthreads = mp.cpu_count()
        else:
            self.nthreads = nthreads
        print('Running with {0} parallel threads'.format(self.nthreads))
        from .sampler import HamiltonianMonteCarloSampler, MetropolisHastingsSampler
        from .NestedSampling import NestedSampler
        from .proposal import DefaultProposalCycle, HamiltonianProposalCycle
        self.user     = usermodel
        self.verbose  = verbose
        self.output   = output
        self.poolsize = poolsize
        self.posterior_samples = None
        
        if seed is None: self.seed=1234
        else:
            self.seed=seed
        
        self.process_pool = []
        # set up the communication pipes
        self.producer_pipes, self.consumer_pipes = self.set_communications()
        # instantiate the nested sampler class
        if resume_file is None:
            self.NS = NestedSampler(self.user,
                        nlive          = nlive,
                        output         = output,
                        verbose        = verbose,
                        seed           = self.seed,
                        prior_sampling = False,
                        resume_file    = None)
        else:
            self.NS = NestedSampler.resume(resume_file)

        # instantiate the sampler class
        for i in range(self.nthreads-nhamiltonian):
            if resume_file is None:
                sampler = MetropolisHastingsSampler(self.user,
                                  maxmcmc,
                                  verbose     = verbose,
                                  output      = output,
                                  poolsize    = poolsize,
                                  seed        = self.seed+i,
                                  proposal    = DefaultProposalCycle(),
                                  resume_file = None
                                  )
            else:
                sampler = MetropolisHastingsSampler.resume(resume_file)

            p = mp.Process(target=sampler.produce_sample, args=(self.producer_pipes[i], self.NS.logLmin, ))
            self.process_pool.append(p)
        
        for i in range(self.nthreads-nhamiltonian,self.nthreads):
            if resume_file is None:
                sampler = HamiltonianMonteCarloSampler(self.user,
                                  maxmcmc,
                                  verbose     = verbose,
                                  output      = output,
                                  poolsize    = poolsize,
                                  seed        = self.seed+i,
                                  proposal    = HamiltonianProposalCycle(model=self.user),
                                  resume_file = None
                                  )
            else:
                sampler = HamiltonianMonteCarloSampler.resume(resume_file)
            p = mp.Process(target=sampler.produce_sample, args=(self.producer_pipes[i], self.NS.logLmin, ))
            self.process_pool.append(p)

    def set_communications(self):
        """
        Sets up the `multiprocessing.Pipe` communication
        channels
        """
        producer_pipes = []
        consumer_pipes = []
        for i in range(self.nthreads):
            consumer, producer = mp.Pipe(duplex=True)
            producer_pipes.append(producer)
            consumer_pipes.append(consumer)
        return producer_pipes, consumer_pipes

    def run(self):
        """
        Run the sampler
        """
        for each in self.process_pool:
            each.start()
        
        self.NS.nested_sampling_loop(self.consumer_pipes)

        for each in self.process_pool:
            each.join()

        self.posterior_samples = self.get_posterior(filename=None)
        if self.verbose>1: self.plot()


    def get_posterior(self, filename='posterior.dat'):
        """
        Returns posterior samples

        Parameters
        ----------
        filename : string
            File to save posterior to

        Returns
        -------
        pos : `np.ndarray`
        """
        import numpy as np
        import os
        from .nest2pos import draw_posterior_many
        import numpy.lib.recfunctions as rfn
        self.nested_samples = rfn.stack_arrays(
                [self.NS.nested_samples[j].asnparray()
                    for j in range(len(self.NS.nested_samples))]
                ,usemask=False)
        posterior_samples = draw_posterior_many([self.nested_samples],[self.NS.Nlive],verbose=self.verbose)
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
        Make some plots of the posterior and nested samples
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

    def worker_sampler(self,*args):
        cProfile.runctx('self.Evolver.produce_sample(*args)', globals(), locals(), 'prof_sampler.prof')
    
    def worker_ns(self,*args):
        cProfile.runctx('self.NS.nested_sampling_loop(*args)', globals(), locals(), 'prof_nested_sampling.prof')

    def profile(self):
        for i in range(0,self.NUMBER_OF_PRODUCER_PROCESSES):
            p = mp.Process(target=self.worker_sampler, args=(self.queues[i%len(self.queues)], self.NS.logLmin ))
            self.process_pool.append(p)
        for i in range(0,self.NUMBER_OF_CONSUMER_PROCESSES):
            p = mp.Process(target=self.worker_ns, args=(self.queues, self.port, self.authkey))
            self.process_pool.append(p)
        for each in self.process_pool:
            each.start()
