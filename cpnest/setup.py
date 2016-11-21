#! /usr/bin/env python
# coding: utf-8

import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue
import cpnest
from cpnest import Sampler,NestedSampling
from cpnest.Sampler import Sampler
from cpnest.NestedSampling import NestedSampler

class setup(object):
    def __init__(self,userclass,Nlive=100,output='./',verbose=0,seed=None,maxmcmc=100,Nthreads=1):
        self.user=userclass
        if seed is None: self.seed=1234
        else:
            self.seed=seed
        self.NS = NestedSampler(self.user,Nlive=Nlive,output=output,verbose=verbose,seed=self.seed,prior=False)
        self.Evolver = Sampler(self.user,maxmcmc,verbose=0)
        self.NUMBER_OF_PRODUCER_PROCESSES = Nthreads
        self.NUMBER_OF_CONSUMER_PROCESSES = 1

        self.process_pool = []
        self.ns_lock = Lock()
        self.sampler_lock = Lock()
        self.queue = Queue()
        self.port=5555
        self.authkey = "12345"
        self.ip = "0.0.0.0"


    def run(self):
        for i in xrange(0,self.NUMBER_OF_PRODUCER_PROCESSES):
            p = Process(target=self.Evolver.produce_sample, args=(self.ns_lock, self.queue, self.NS.jobID, self.NS.logLmin, self.seed+i, self.ip, self.port, self.authkey ))
            self.process_pool.append(p)
        for i in xrange(0,self.NUMBER_OF_CONSUMER_PROCESSES):
            p = Process(target=self.NS.nested_sampling_loop, args=(self.sampler_lock, self.queue, self.port, self.authkey))
            self.process_pool.append(p)
        for each in self.process_pool:
            each.start()


if __name__ == '__main__':
    import optparse as op
    parser = op.OptionParser()
    parser.add_option("-N", type="int", dest="Nlive", help="Number of Live points",default=1000)
    parser.add_option("-o", "--output", type="string", dest="output", help="Output folder", default=None)
    parser.add_option("-i", "--input", type="string", dest="input", help="Input folder", default=None)
    parser.add_option("-s", type="int", dest="seed", help="seed for the chain", default=0)
    parser.add_option("--verbose", type="int", dest="verbose", help="display progress information", default=1)
    parser.add_option("--maxmcmc", type="int", dest="maxmcmc", help="maximum number of mcmc steps", default=5000)
    parser.add_option("--nthreads", type="int", dest="nthreads", help="number of sampling threads to spawn", default=None)
    parser.add_option( "--sample-prior", action="store_true", dest="prior", help="draw NLIVE samples from the prior", default=False)
    
    (options, args) = parser.parse_args()

    verbose_ns = False
    verbose_sam = False
    
    if options.verbose == 1:
        verbose_ns = True
    elif options.verbose == 2:
        verbose_ns = True
        verbose_sam = True

    port = 5555
    authkey = "12345"
    ip = "0.0.0.0"
    
    np.random.seed(options.seed)
    if options.input is not None:
        data = np.loadtxt(options.input)
    else:
        data = [x for x in np.random.normal(0.5,0.5,size=1000)]
    bounds = [[0.0,1.0],[0.0,1.0]]
    names = ['mean','sigma']

    out_folder = options.output

    os.system('mkdir -p %s'%(out_folder))
    NS = NestedSampler(data,names,bounds,Nlive=options.Nlive,maxmcmc=options.maxmcmc,output=out_folder,verbose=verbose_ns,seed=options.seed,prior=options.prior)
    Evolver = Sampler(data,options.maxmcmc,names,bounds, verbose = verbose_sam)

    NUMBER_OF_PRODUCER_PROCESSES = options.nthreads
    NUMBER_OF_CONSUMER_PROCESSES = 1

    process_pool = []
    ns_lock = Lock()
    sampler_lock = Lock()
    queue = Queue()
    
    for i in xrange(0,NUMBER_OF_PRODUCER_PROCESSES):
        p = Process(target=Evolver.produce_sample, args=(ns_lock, queue, NS.jobID, NS.logLmin, options.seed+i,ip, port, authkey ))
        process_pool.append(p)
    for i in xrange(0,NUMBER_OF_CONSUMER_PROCESSES):
        p = Process(target=NS.nested_sampling_loop, args=(sampler_lock, queue, port, authkey))
        process_pool.append(p)
    for each in process_pool:
        each.start()
