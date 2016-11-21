from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport log,exp
import sys
import os
import cPickle as pickle
from collections import deque
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue
import Queue as QQueue
from multiprocessing.sharedctypes import Value
import parameter
from ctypes import c_int, c_double
import types
import copy_reg

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

cdef inline double log_add(double x, double y): return x+log(1+exp(y-x)) if x >= y else y+log(1+exp(x-y))

class NestedSampler(object):
    """
    
    Initialisation arguments:
    
    Nlive: 
        number of live points to be used for the integration
        default: 1024
    
    maxmcmc: 
        maximum number of mcmc steps to be used in the sampler
        default: 4096
    
    output: 
        folder where the output will be stored
    
    verbose: 
        display information on screen
        default: True
        
    seed:
        seed for the initialisation of the pseudorandom chain
    
    prior:
        produce N samples from the prior

    """

    def __init__(self,data,names,bounds,Nlive=1024,maxmcmc=4096,model=None,events=None,galaxies=None,output=None,verbose=True,seed=1,prior=False):
        """
        Initialise all necessary arguments and variables for the algorithm
        """

        self.prior_sampling = prior
        self.setup_random_seed(seed)
        self.verbose = verbose
        self.active_index = 0
        self.accepted = 0
        self.rejected = 1
        self.dimension = 0
        self.Nlive = Nlive
        self.Nmcmc = maxmcmc
        self.maxmcmc = maxmcmc
        self.events = events
        self.galaxies = galaxies
        self.output,self.evidence_out,self.checkpoint = self.setup_output(output)
        self.data = data
        self.params = [None] * self.Nlive
        self.logZ = np.finfo(np.float128).min
        self.tolerance = 0.01
        self.condition = np.inf
        self.information = 0.0
        self.worst = 0
        self.logLmax = -np.inf
        self.iteration = 0
        self.nextID = 0
        self.samples_cache = {}
        for n in range(self.Nlive):
            while True:
                if self.verbose: sys.stderr.write("sprinkling %d live points --> %.3f %% complete\r"%(self.Nlive,100.0*float(n+1)/float(self.Nlive)))
                self.params[n] = parameter.LivePoint(names,bounds)
                self.params[n].logP = parameter.logPrior(self.data,self.params[n])
                self.params[n].logL = parameter.logLikelihood(self.data,self.params[n])
                if not(np.isinf(self.params[n].logP)) and not(np.isinf(self.params[n].logL)): break
        sys.stderr.write("\n")
        self.dimension = self.params[0].dimension

        self.active_live = parameter.LivePoint(names,bounds)

#        if self.loadState()==0:
#            sys.stderr.write("Loaded state %s, resuming run\n"%self.checkpoint)
#            self.new_run=False
#        else:
#            sys.stderr.write("Checkpoint not found, starting anew\n")
#            self.new_run=True

        if self.verbose: sys.stderr.write("Dimension --> %d\n"%self.dimension)
        header = open(os.path.join(output,'header.txt'),'w')
        for n in self.active_live.names:
            header.write(n+'\t')
        header.write('logL\n')
        header.close()
        self.jobID = Value(c_int,0,lock=Lock())
        self.logLmin = Value(c_double,-np.inf,lock=Lock())

    def setup_output(self,output):
        """
        Set up the output folder
        """
        os.system("mkdir -p %s"%output)
        outputfile = "chain_"+str(self.Nlive)+"_"+str(self.seed)+".txt"
        return open(os.path.join(output,outputfile),"w"),open(os.path.join(output,outputfile+"_evidence.txt"), "wb" ),os.path.join(output,outputfile+"_resume")

    def setup_random_seed(self,seed):
        """
        initialise the random seed
        """
        self.seed = seed
        np.random.seed(seed=self.seed)

    def _select_live(self):
        """
        select a live point
        """
        while True:
            j = np.random.randint(self.Nlive)
            if j!= self.worst:
                return j

    def consume_sample(self, producer_lock, queue, port, authkey):

        while not(self.nextID in self.samples_cache):
            ID,acceptance,jumps,values,logP,logL = queue.get()
            self.samples_cache[ID] = acceptance,jumps,values,logP,logL

        acceptance,jumps,values,logP,logL = self.samples_cache.pop(self.nextID)
        self.rejected+=1
        self.nextID += 1

        for j in range(self.params[self.worst].dimension):
            self.params[self.worst].parameters[j].value = np.copy(values[j])
        
        self.params[self.worst].logP = np.copy(logP)
        self.params[self.worst].logL = np.copy(logL)
        if self.params[self.worst].logL>self.logLmin.value:

            logLmin = self.get_worst_live_point()

            logWt = logLmin+self.logwidth;
            logZnew = log_add(self.logZ, logWt)
            self.information = exp(logWt - logZnew) * self.params[self.worst].logL + exp(self.logZ - logZnew) * (self.information + self.logZ) - logZnew
            self.logZ = logZnew
            self.condition = log_add(self.logZ,self.logLmax-self.iteration/(float(self.Nlive))-self.logZ)
            line = ""
            for n in self.params[self.worst].names:
                line+='%.30e\t'%self.params[self.worst].get(n)
            line+='%30e\n'%self.params[self.worst].logL
            self.output.write(line)
            self.active_index=self._select_live()
            parameter.copy_live_point(self.params[self.worst],self.params[self.active_index])
            if self.verbose: sys.stderr.write("%d: n:%4d acc:%.3f H: %.2f logL %.5f --> %.5f dZ: %.3f logZ: %.3f logLmax: %.5f cache: %d\n"%(self.iteration,jumps,acceptance,self.information,logLmin,self.params[self.worst].logL,self.condition,self.logZ,self.logLmax,len(self.samples_cache)))
            self.logwidth-=1.0/float(self.Nlive)
            self.iteration+=1

    def get_worst_live_point(self):
        logL_array = np.array([p.logL for p in self.params])
        self.worst = logL_array.argmin()
        self.logLmin.value = np.min(logL_array)
        self.logLmax = np.max(logL_array)
        return np.float128(self.logLmin.value)

    def nested_sampling_loop(self, producer_lock, queue, port, authkey):
        """
        main nested sampling loop
        """
        self.logwidth = log(1.0 - exp(-1.0 / float(self.Nlive)))
        for i in range(self.Nlive):
            while True:
                while not(self.nextID in self.samples_cache):
                    ID,acceptance,jumps,values,logP,logL = queue.get()
                    self.samples_cache[ID] = acceptance,jumps,values,logP,logL
                acceptance,jumps,values,logP,logL = self.samples_cache.pop(self.nextID)
                self.nextID +=1
                for j in range(self.params[i].dimension):
                    self.params[i].parameters[j].value = np.copy(values[j])
                self.params[i].logP = np.copy(logP)
                self.params[i].logL = np.copy(logL)
                if self.params[i].logP!=-np.inf or self.params[i].logL!=-np.inf: break
            if self.verbose: sys.stderr.write("sampling the prior --> %.3f %% complete\r"%(100.0*float(i+1)/float(self.Nlive)))
        if self.verbose: sys.stderr.write("\n")
        if self.prior_sampling:
            for i in range(self.Nlive):
                line = ""
                for n in self.params[i].names:
                    line+='%.30e\t'%self.params[i].get(n)
                line+='%30e\n'%self.params[i].logL
                self.output.write(line)
            self.output.close()
            self.evidence_out.close()
            self.logLmin.value = 999
            sys.stderr.write("Nested Sampling process %s, exiting\n"%os.getpid())
            return 0
        
        logLmin = self.get_worst_live_point()

        logWt = logLmin+self.logwidth;
        logZnew = log_add(self.logZ, logWt)
        self.information = exp(logWt - logZnew) * self.params[self.worst].logL + exp(self.logZ - logZnew) * (self.information + self.logZ) - logZnew
        self.logZ = logZnew
        self.condition = log_add(self.logZ,self.logLmax-self.iteration/(float(self.Nlive))-self.logZ)
        line = ""
        for n in self.params[self.worst].names:
            line+='%.30e\t'%self.params[self.worst].get(n)
        line+='%30e\n'%self.params[self.worst].logL
        self.output.write(line)
        self.active_index =self._select_live()
        parameter.copy_live_point(self.params[self.worst],self.params[self.active_index])

        while self.condition > self.tolerance:
            self.consume_sample(producer_lock, queue, port, authkey)

        sys.stderr.write("\n")

        # final adjustments
        self.logLmin.value = 999
        while len(self.samples_cache) > 0:
            self.consume_sample(producer_lock, queue, port, authkey)
            self.logLmin.value = 999

        i = 0
        logL_array = [p.logL for p in self.params]
        logL_array = np.array(logL_array)
        idx = logL_array.argsort()
        logL_array = logL_array[idx]
        for i in idx:
            line = ""
            for n in self.params[i].names:
                line+='%.30e\t'%self.params[i].get(n)
            line+='%30e\n'%self.params[i].logL
            self.output.write(line)
            i+=1
        self.output.close()
        self.evidence_out.write('%.5f %.5f\n'%(self.logZ,self.logLmax))
        self.evidence_out.close()
        k = 0
        sys.stderr.write("Nested Sampling process %s, exiting\n"%os.getpid())
        return 0

#    def saveState(self):
#        try:
#            livepoints_stack = np.zeros(self.Nlive,dtype={'names':self.params[0].par_names,'formats':self.params[0].par_types})
#            for i in xrange(self.Nlive):
#                for n in self.params[0].par_names:
#                    livepoints_stack[i][n] = self.params[i]._internalvalues[n]
#            resume_out = open(self.checkpoint,"wb")
#            pickle.dump((livepoints_stack,np.random.get_state(),self.iteration,self.cache),resume_out)
#            sys.stderr.write("Checkpointed %d live points.\n"%self.Nlive)
#            resume_out.close()
#            return 0
#        except:
#            sys.stderr.write("Checkpointing failed!\n")
#            return 1
#
#    def loadState(self):
#        try:
#            resume_in = open(self.checkpoint,"rb")
#            livepoints_stack,RandomState,self.iteration,self.cache = pickle.load(resume_in)
#            resume_in.close()
#            for i in xrange(self.Nlive):
#                for n in self.params[0].par_names:
#                    self.params[i]._internalvalues[n] = livepoints_stack[i][n]
#                self.params[i].logPrior()
#                self.params[i].logLikelihood()
#            np.random.set_state(RandomState)
#            self.kwargs=proposals._setup_kwargs(self.params,self.Nlive,self.dimension)
#            self.kwargs=proposals._update_kwargs(**self.kwargs)
#            sys.stderr.write("Resumed %d live points.\n"%self.Nlive)
#            return 0
#        except:
#            sys.stderr.write("Resuming failed!\n")
#            return 1


def parse_to_list(option, opt, value, parser):
    """
    parse a comma separated string into a list
    """
    setattr(parser.values, option.dest, value.split(','))

if __name__=="__main__":
    pass
