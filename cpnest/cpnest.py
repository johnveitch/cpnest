#! /usr/bin/env python
# coding: utf-8

import sys
import os
import optparse as op
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue
from Sampler import *
from ctypes import c_int, c_double
from NestedSampling import *
from parameter import *
#import matplotlib.cm as cm

def compute_rate(dl_list,T):
    Dmax = np.median([e.dmax for e in dl_list])
    (idx,) = np.where([e.dl for e in dl_list] < Dmax)
    Vmax = (4.*np.pi*Dmax**3)/3.0
    return len(idx)/(Vmax*T)

def FindHeightForLevel(inArr, adLevels):
    # flatten the array
    oldshape = shape(inArr)
    adInput= reshape(inArr,oldshape[0]*oldshape[1])
    # GET ARRAY SPECIFICS
    nLength = np.size(adInput)
      
    # CREATE REVERSED SORTED LIST
    adTemp = -1.0 * adInput
    adSorted = np.sort(adTemp)
    adSorted = -1.0 * adSorted

    # CREATE NORMALISED CUMULATIVE DISTRIBUTION
    adCum = np.zeros(nLength)
    adCum[0] = adSorted[0]
    for i in xrange(1,nLength):
        adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])
    adCum = adCum - adCum[-1]

    # FIND VALUE CLOSEST TO LEVELS
    adHeights = []
    for item in adLevels:
        idx=(np.abs(adCum-np.log(item))).argmin()
        adHeights.append(adSorted[idx])

    adHeights = np.array(adHeights)
    return adHeights

if __name__ == '__main__':
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
