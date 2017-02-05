import sys
import os
import numpy as np
from collections import deque
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue

from . import parameter
from . import proposals

class Sampler(object):
    """
    Sampler class.
    Initialisation arguments:
    
    usermodel:
    user defined model to sample
    
    maxmcmc:
    maximum number of mcmc steps to be used in the sampler
    default: 4096
    
    verbose:
    display debug information on screen
    default: False
    
    poolsize:
    number of objects for the affine invariant sampling
    default: 100
    
    """
    def __init__(self,usermodel,maxmcmc,verbose=False,poolsize=100):
        self.user = usermodel
        names = usermodel.par_names
        bounds = usermodel.bounds
        self.cache = deque(maxlen=maxmcmc)
        self.maxmcmc = maxmcmc
        self.Nmcmc = maxmcmc
        self.proposals = proposals.setup_proposals_cycle()
        self.poolsize = poolsize
        #self.evolution_points = [None]*self.poolsize
        self.evolution_points = deque(maxlen=self.poolsize)
        self.verbose=verbose
        self.inParam = parameter.LivePoint(names,bounds)
        self.param = parameter.LivePoint(names,bounds)
        self.dimension = self.param.dimension
        for n in range(self.poolsize):
          while True:
            if self.verbose: sys.stderr.write("process {0!s} --> generating pool of {1:d} points for evolution --> {2:.3f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(n+1)/float(self.poolsize)))
            p = parameter.LivePoint(names,bounds)
            p.initialise()
            p.logP = self.user.log_prior(p)
            if np.isfinite(p.logP): break
          p.logL=self.user.log_likelihood(p)
          self.evolution_points.append(p)
        if self.verbose: sys.stderr.write("\n")
        self.kwargs = proposals.ProposalArguments(self.dimension)
        self.kwargs.update(list(self.evolution_points))
        for _ in range(self.poolsize):
          s = self.evolution_points.popleft()
          acceptance,jumps,s = self.metropolis_hastings(s,-np.inf,self.Nmcmc)
          self.evolution_points.append(s)
        self.kwargs.update(list(self.evolution_points))

    def produce_sample(self, consumer_lock, queue, IDcounter, logLmin, seed, ip, port, authkey):
        """
        main loop that generates samples and puts them in the queue for the nested sampler object
        """
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.counter=0

        while(1):
            IDcounter.get_lock().acquire()
            job_id = IDcounter.get_obj()
            id = job_id.value
            job_id.value+=1
            IDcounter.get_lock().release()
            if logLmin.value==np.inf:
                break
            acceptance,jumps,outParam = self.metropolis_hastings(self.inParam,logLmin.value,self.Nmcmc)
            self.evolution_points.append(outParam.copy())
            #queue.put((id,acceptance,jumps,np.array([outParam[n] for n in outParam.names]),outParam.logP,outParam.logL))
            queue.put((id,acceptance,jumps,outParam))
            if self.counter == 0 and len(self.cache)==5*self.maxmcmc:
                self.autocorrelation()
                self.kwargs.update(list(self.evolution_points))
            elif (self.counter%(self.poolsize/2))==0 or acceptance<0.01:
                self.autocorrelation()
                self.kwargs.update(list(self.evolution_points))
            self.counter += 1
        sys.stderr.write("Sampler process {0!s}, exiting\n".format(os.getpid()))
        return 0

    def metropolis_hastings(self,inParam,logLmin,nsteps):
        """
        metropolis-hastings loop to generate the new live point taking nmcmc steps
        """
        accepted = 0
        rejected = 1
        jumps = 0
        oldparam=inParam.copy()
        logp_old = self.user.log_prior(oldparam)
        while (jumps < nsteps or accepted==0):
            newparam,log_acceptance = self.proposals[jumps%100].get_sample(oldparam.copy(),list(self.evolution_points),self.kwargs)
            newparam.logP = self.user.log_prior(newparam)
            if newparam.logP-logp_old > log_acceptance:
                newparam.logL = self.user.log_likelihood(newparam)
                if newparam.logL > logLmin:
                  oldparam=newparam.copy()
                  logp_old = newparam.logP
                  accepted+=1
                else:
                  rejected+=1
            else:
                rejected+=1

            self.cache.append(np.array(oldparam.values))
            jumps+=1
            if jumps==10*self.maxmcmc:
              print('Warning, MCMC chain exceeded {0} iterations!'.format(10*self.maxmcmc))
        return (float(accepted)/float(rejected+accepted),jumps,oldparam)

    def autocorrelation(self):
        """
        estimates the autocorrelation length of the mcmc chain from the cached samples
        """
        try:
            ACLs = []
            cov_array = np.array(self.cache).T
            N =  cov_array.shape[1]
            for i,n in enumerate(self.evolution_points[0].names):
                ACF = proposals.autocorrelation(cov_array[i,:])
                ACL = np.min(np.where((ACF > -2./np.sqrt(N)) & (ACF < 2./np.sqrt(N)))[0])
                if not(np.isnan(ACL)):
                    ACLs.append(ACL)
                    if self.verbose: sys.stderr.write("process {0!s} --> cache size: {1:d} autocorrelation length {2!s} = {3:.1f} mean = {4:g} standard deviation = {5:g}\n".format(os.getpid(), len(self.cache), n, ACLs[-1], np.mean(cov_array[:,i]), np.std(cov_array[:,i])))
            self.Nmcmc =int((np.max(ACLs)))
            if self.Nmcmc < 2: self.Nmcmc = 2
            if self.Nmcmc > self.maxmcmc:
                sys.stderr.write("Warning ACL --> {0:d}!\n".format(self.Nmcmc))
                self.Nmcmc = self.maxmcmc
        except:
            sys.stderr.write("Warning ACL failed! setting {0:d}!\n".format(self.maxmcmc))
            self.Nmcmc = self.maxmcmc
        if len(self.cache)==5*self.maxmcmc:
            self.cache = deque(maxlen=5*self.maxmcmc)

if __name__=="__main__":
    pass
