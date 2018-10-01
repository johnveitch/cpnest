#! /usr/bin/env python
# coding: utf-8

import multiprocessing as mp
import cProfile
import time
import os

class CPNest(object):
    """
    Class to control CPNest sampler
    cp = CPNest(usermodel,Nlive=100,output='./',verbose=0,seed=None,maxmcmc=100,Nthreads=None,balanced_sampling = True)
    
    Input variables:
    usermodel : an object inheriting cpnest.model.Model that defines the user's problem
    Nlive : Number of live points (100)
    Poolsize: Number of objects in the sampler pool (100)
    output : output directory (./)
    verbose: Verbosity, 0=silent, 1=progress, 2=diagnostic, 3=detailed diagnostic
    seed: random seed (default: 1234)
    maxmcmc: maximum MCMC points for sampling chains (100)
    Nthreads: number of parallel samplers. Default (None) uses mp.cpu_count() to autodetermine
    balance_samplers: If False, more samples will come from threads sampling "fast" parts of parameter space.
                      This may cause bias if parts of your parameter space are more expensive than others.
                      Default: True    
    """
    def __init__(self,usermodel,Nlive=100,Poolsize=100,output='./',verbose=0,seed=None,maxmcmc=100,Nthreads=None,balance_samplers = True):
        if Nthreads is None:
            Nthreads = mp.cpu_count()
        print('Running with {0} parallel threads'.format(Nthreads))
        from .sampler import HamiltonianMonteCarloSampler, MetropolisHastingsSampler
        from .NestedSampling import NestedSampler
        from .proposal import DefaultProposalCycle, HamiltonianProposalCycle
        self.user=usermodel
        self.verbose=verbose
        self.output=output
        self.poolsize = Poolsize
        if seed is None: self.seed=1234
        else:
            self.seed=seed
        
        self.NS = NestedSampler(self.user,Nlive=Nlive,output=output,verbose=verbose,seed=self.seed,prior_sampling=False)

        self.process_pool = []

        self.consumer_pipes = []
        
        for i in range(Nthreads):
            sampler = HamiltonianMonteCarloSampler(self.user,
                              maxmcmc,
                              verbose=verbose,
                              output=output,
                              poolsize=Poolsize,
                              seed=self.seed+i,
                              proposal = HamiltonianProposalCycle(model=self.user)
                              )
            # We set up pipes between the nested sampling and the various sampler processes
            consumer, producer = mp.Pipe(duplex=True)
            self.consumer_pipes.append(consumer)
            p = mp.Process(target=sampler.produce_sample, args=(producer, self.NS.logLmin, ))
            self.process_pool.append(p)

    def run(self):
        """
        Run the sampler
        """
        import numpy as np
        import os
        from .nest2pos import draw_posterior_many, redraw_mcmc_chain

        for each in self.process_pool:
            each.start()
        
        self.NS.nested_sampling_loop(self.consumer_pipes)

        for each in self.process_pool:
            each.join()

        import numpy.lib.recfunctions as rfn
        self.nested_samples = rfn.stack_arrays([self.NS.nested_samples[j].asnparray() for j in range(len(self.NS.nested_samples))],usemask=False)
        if self.verbose>=3:
            
            chain = [redraw_mcmc_chain(np.genfromtxt(os.path.join(self.NS.output_folder,'mcmc_chain_%d.dat'%each.pid), names=True),verbose=self.verbose) for each in self.process_pool]
            self.posterior_samples = rfn.stack_arrays(chain)
            nssamps = draw_posterior_many([self.nested_samples],[self.NS.Nlive],verbose=self.verbose)
            self.posterior_samples = rfn.stack_arrays([self.posterior_samples,nssamps])
        else:
            self.posterior_samples = draw_posterior_many([self.nested_samples],[self.NS.Nlive],verbose=self.verbose)
        self.posterior_samples = np.array(self.posterior_samples)
        np.savetxt(os.path.join(self.NS.output_folder,'posterior.dat'),self.posterior_samples.ravel(),header=' '.join(self.posterior_samples.dtype.names),newline='\n',delimiter=' ')
        if self.verbose>1: self.plot()

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
