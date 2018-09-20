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
    proposal: cpnest.proposal.ProposalCycle object to use while sampling.
    """
    def __init__(self,usermodel,Nlive=100,Poolsize=100,output='./',verbose=0,seed=None,maxmcmc=100,Nthreads=None,balance_samplers = True, proposal=None):
        if Nthreads is None:
            Nthreads = mp.cpu_count()
        print('Running with {0} parallel threads'.format(Nthreads))
        from .sampler import Sampler
        from .NestedSampling import NestedSampler
        self.user=usermodel
        self.verbose=verbose
        self.output=output
        if seed is None: self.seed=1234
        else:
            self.seed=seed
        
        self.NS = NestedSampler(self.user,Nlive=Nlive,output=output,verbose=verbose,seed=self.seed,prior_sampling=False)

        self.process_pool = []

        self.consumer_pipes = []
        for i in range(Nthreads):
            sampler = Sampler(self.user,maxmcmc,verbose=verbose,output=output,poolsize=Poolsize,seed=self.seed+i, proposal=proposal)
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
        from .nest2pos import draw_posterior_many

        for each in self.process_pool:
            each.start()
        
        self.NS.nested_sampling_loop(self.consumer_pipes)

        for each in self.process_pool:
            each.join()

        import numpy.lib.recfunctions as rfn
        self.nested_samples = rfn.stack_arrays([self.NS.nested_samples[j].asnparray() for j in range(len(self.NS.nested_samples))],usemask=False)
        self.posterior_samples = draw_posterior_many([self.nested_samples],[self.NS.Nlive],verbose=self.verbose)
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

    def worker_sampler(self, producer_pipe, logLmin):
        cProfile.runctx('self.sampler.produce_sample(producer_pipe, logLmin)', globals(), locals(), 'prof_sampler.prof')
    
    def worker_ns(self):
        cProfile.runctx('self.NS.nested_sampling_loop(self.consumer_pipes)', globals(), locals(), 'prof_nested_sampling.prof')

    def profile(self):
        for i in range(0,self.Nthreads):
            p = mp.Process(target=self.worker_sampler, args=(self.producer_pipes[i], self.NS.logLmin, ))
            p.start()
        p = mp.Process(target=self.worker_ns)
        p.start()
