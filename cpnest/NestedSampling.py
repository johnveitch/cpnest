from __future__ import division, print_function
import sys
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Lock
from numpy import logaddexp, exp
from numpy import inf
from math import isnan
try:
    from queue import Empty
except ImportError:
    from Queue import Empty # For python 2 compatibility
from multiprocessing.sharedctypes import Value
from ctypes import c_int, c_double
import types
from . import nest2pos
from .nest2pos import logsubexp
from operator import attrgetter

try:
    import copyreg
except ImportError:
    import copy_reg as copyreg


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)

class _NSintegralState(object):
  """
  Stores the state of the nested sampling integrator
  """
  def __init__(self,nlive):
    self.nlive=nlive
    self.reset()
  def reset(self):
    """
    Reset the sampler
    """
    self.iteration=0
    self.logZ=-inf
    self.oldZ=-inf
    self.logw=0
    self.info=0
    # Start with a dummy sample enclosing the whole prior
    self.logLs=[-inf] # Likelihoods sampled
    self.log_vols=[0.0] # Volumes enclosed by contours
  def increment(self,logL,nlive=None):
    """
    Increment the state of the evidence integrator
    Simply uses rectangle rule for initial estimate
    """
    if(logL<=self.logLs[-1]):
      print('WARNING: NS integrator received non-monotonic logL. {0:.3f} -> {1:.3f}'.format(self.logLs[-1],logL))
    if nlive is None:
      nlive = self.nlive
    oldZ = self.logZ
    logt=-1.0/nlive
    Wt = self.logw + logL + logsubexp(0,logt)
    self.logZ = logaddexp(self.logZ,Wt)
    # Update information estimate
    if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
        self.info = np.exp(Wt - self.logZ)*logL + np.exp(oldZ - self.logZ)*(self.info + oldZ) - self.logZ
        if isnan(self.info):
            self.info=0
    
    # Update history
    self.logw += logt
    self.iteration += 1
    self.logLs.append(logL)
    self.log_vols.append(self.logw)
  def finalise(self):
    """
    Compute the final evidence with more accurate integrator
    Call at end of sampling run to refine estimate
    """
    from scipy import integrate
    # Trapezoidal rule
    self.logZ=nest2pos.log_integrate_log_trap(np.array(self.logLs),np.array(self.log_vols))
    return self.logZ
  def plot(self,filename):
    """
    Plot the logX vs logL
    """
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pyplot as plt
    fig=plt.figure()
    plt.plot(self.log_vols,self.logLs)
    plt.title('{0} iterations. logZ={1:.2f} H={2:.2f} bits'.format(self.iteration,self.logZ,self.info*np.log2(np.e)))
    plt.grid(which='both')
    plt.xlabel('log prior_volume')
    plt.ylabel('log likelihood')
    plt.xlim([self.log_vols[-1],self.log_vols[0]])
    plt.savefig(filename)
    print('Saved nested sampling plot as {0}'.format(filename))


class NestedSampler(object):
    """
    Nested Sampler class.
    Initialisation arguments:
    
    Nlive:
        number of live points to be used for the integration
        default: 1024
    
    maxmcmc:
        maximum number of mcmc steps to be used in the sampler
        default: 4096
    
    output:
        folder where the output will be stored
    
    verbose: Verbosity level
	0: Nothing
        1: display information on screen
        2: (1) + diagnostic plots
        
    seed:
        seed for the initialisation of the pseudorandom chain
    
    prior_sampling:
        produce Nlive samples from the prior
        
    stopping:
	Stop when remaining samples wouldn't change logZ estimate by this much
    
    """

    def __init__(self,usermodel,Nlive=1024,maxmcmc=4096,output=None,verbose=1,seed=1,prior_sampling=False,stopping=0.1):
        """
        Initialise all necessary arguments and variables for the algorithm
        """
        self.model=usermodel
        self.prior_sampling = prior_sampling
        self.setup_random_seed(seed)
        self.verbose = verbose
        self.accepted = 0
        self.rejected = 1
        self.queue_counter = 0
        self.Nlive = Nlive
        self.Nmcmc = maxmcmc
        self.maxmcmc = maxmcmc
        self.params = [None] * self.Nlive
        self.tolerance = stopping
        self.condition = np.inf
        self.worst = 0
        self.logLmax = -np.inf
        self.iteration = 0
        self.nested_samples=[]
        self.logZ=None
        self.state = _NSintegralState(self.Nlive)
        sys.stdout.flush()
        self.output_folder = output
        self.output,self.evidence_out,self.checkpoint = self.setup_output(output)
        header = open(os.path.join(output,'header.txt'),'w')
        header.write('\t'.join(self.model.names))
        header.write('\tlogL\n')
        header.close()
        self.logLmin = Value(c_double,-np.inf,lock=Lock())

    def setup_output(self,output):
        """
        Set up the output folder
        """
        os.system("mkdir -p {0!s}".format(output))
        self.outfilename = "chain_"+str(self.Nlive)+"_"+str(self.seed)+".txt"
        self.outputfile=open(os.path.join(output,self.outfilename),"w")
        self.outputfile.write('{0:s}\n'.format(self.model.header().rstrip()))
        self.outputfile.flush()
        return self.outputfile,open(os.path.join(output,self.outfilename+"_evidence.txt"), "w" ),os.path.join(output,self.outfilename+"_resume")

    def output_sample(self,sample):
        self.outputfile.write('{0:s}\n'.format(self.model.strsample(sample).rstrip()))
        self.outputfile.flush()
        self.nested_samples.append(sample)

    def setup_random_seed(self,seed):
        """
        initialise the random seed
        """
        self.seed = seed
        np.random.seed(seed=self.seed)

    def consume_sample(self, consumer_pipes):
        """
        consumes a sample from the shared queues and updates the evidence
        """
        # Increment the state of the evidence integration
        logLmin = self.get_worst_n_live_points(len(consumer_pipes))
        logLtmp = []
        for k in self.worst:
            self.state.increment(self.params[k].logL)
            consumer_pipes[k].send(self.params[k])
            self.output_sample(self.params[k])
            logLtmp.append(self.params[k].logL)
        self.condition = logaddexp(self.state.logZ,self.logLmax - self.iteration/(float(self.Nlive))) - self.state.logZ
        
        # Replace the points we just consumed with the next acceptable ones
        for k in self.worst:
            self.iteration+=1
            loops = 0
            while(True):
                loops += 1
                self.acceptance, self.jumps, proposed = consumer_pipes[self.queue_counter].recv()
                self.queue_counter = (self.queue_counter + 1) % len(consumer_pipes)
                if proposed.logL>self.logLmin.value:
                    # replace worst point with new one
                    self.params[k]=proposed
                    break

            if self.verbose:
                sys.stderr.write("{0:d}: n:{1:4d} acc:{2:.3f} sub_acc:{3:.3f} H: {4:.2f} logL {5:.5f} --> {6:.5f} dZ: {7:.3f} logZ: {8:.3f} logLmax: {9:.2f}\n"\
                .format(self.iteration, self.jumps, self.acceptance/float(loops), self.acceptance, self.state.info,\
                  logLtmp[k], self.params[k].logL, self.condition, self.state.logZ, self.logLmax))
                sys.stderr.flush()

    def get_worst_n_live_points(self, n):
        """
        selects the lowest likelihood N live points
        """
        self.params.sort(key=attrgetter('logL'))
        self.worst = np.arange(n)
        self.logLmin.value = np.float128(self.params[n].logL)
        self.logLmax = np.max(self.params[-1].logL)
        return np.float128(self.logLmin.value)

    def nested_sampling_loop(self, consumer_pipes):
        """
        main nested sampling loop
        """
        # send all live points to the samplers for start
        i = 0
        while i < self.Nlive:
            for j in range(len(consumer_pipes)): consumer_pipes[j].send(self.model.new_point())
            for j in range(len(consumer_pipes)):
                while i<self.Nlive:
                    self.acceptance,self.jumps,self.params[i] = consumer_pipes[self.queue_counter].recv()
                    self.queue_counter = (self.queue_counter + 1) % len(consumer_pipes)
                    if self.params[i].logP!=-np.inf and self.params[i].logL!=-np.inf:
                        i+=1
                        break


            if self.verbose:
                sys.stderr.write("sampling the prior --> {0:.0f} % complete\r".format((100.0*float(i+1)/float(self.Nlive))))
                sys.stderr.flush()
        if self.verbose:
            sys.stderr.write("\n")
            sys.stderr.flush()
        if self.prior_sampling:
            for i in range(self.Nlive):
                self.output_sample(self.params[i])
            self.output.close()
            self.evidence_out.close()
            self.logLmin.value = np.inf
            print("Nested Sampling process {0!s}, exiting".format(os.getpid()))
            return 0

        while self.condition > self.tolerance:
            self.consume_sample(consumer_pipes)

	    # Signal worker threads to exit
        self.logLmin.value = np.inf
        for c in consumer_pipes:
            c.send(None)

        # final adjustments
        self.params.sort(key=attrgetter('logL'))
        for i,p in enumerate(self.params):
            self.state.increment(p.logL,nlive=self.Nlive-i)
            self.output_sample(p)
 
        # Refine evidence estimate
        self.state.finalise()
        self.logZ = self.state.logZ

        self.output.close()
        self.evidence_out.write('{0:.5f} {1:.5f}\n'.format(self.state.logZ, self.logLmax))
        self.evidence_out.close()
        print('Final evidence: {0:0.2f}\nInformation: {1:.2f}'.format(self.state.logZ,self.state.info))
        
        # Some diagnostics
        if self.verbose>1 :
          self.state.plot(os.path.join(self.output_folder,self.outfilename+'.png'))
        return self.state.logZ

