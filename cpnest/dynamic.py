from __future__ import division, print_function
import sys
import os
import numpy as np
from numpy import logaddexp, exp, array, log, log1p
from numpy import inf
from . import nest2pos
from .NestedSampling import NestedSampler, _NSintegralState
from .nest2pos import logsubexp, log_integrate_log_trap
from functools import reduce
import itertools as it
from bisect import bisect_left
import time
from ctypes import c_int, c_double
import types
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
from multiprocessing import Lock


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result

    return timed

class DyNest(object):
    """
    Self-contained class for dynamic nested sampling
    on a single machine
    """
    def __init__(self,usermodel, output='./',Nthreads=None, Ninit=100, maxmcmc=1000, verbose=1, Poolsize=100, seed=None):
        self.process_pool=[]
        if seed:
            self.seed=seed
        else:
            self.seed = np.random.randint(2**32-1)
        if Nthreads is None:
            Nthreads = mp.cpu_count()
        print('Running with {0} parallel threads'.format(Nthreads))
        from .sampler import Sampler
        self.dnest = DynamicNestedSampler(usermodel, output=output, verbose=verbose, seed=seed, Ninit=Ninit)
        
        for i in range(Nthreads):
            print('Initialised sampler')
            sampler = Sampler(usermodel,maxmcmc,verbose=verbose,output=output,poolsize=Poolsize,seed=self.seed+i )
            consumer, producer = mp.Pipe(duplex=True)
            p = mp.Process(target=sampler.produce_sample, args=(producer, self.dnest.logLmin, ))
            print('Connecting sampler {0}'.format(i))
            self.dnest.connect_sampler(consumer)
            #producer.close()
            self.process_pool.append(p)
            
            
    def run(self):
        """
        Run the sampler
        """
        import numpy as np
        import os
        from .nest2pos import draw_posterior_many

        for each in self.process_pool:
            print('Starting sampler {0}'.format(each))
            each.start()
        
        self.dnest.nested_sampling_loop()

        for each in self.process_pool:
            each.join()

        import numpy.lib.recfunctions as rfn
        #self.nested_samples = rfn.stack_arrays([self.NS.nested_samples[j].asnparray() for j in range(len(self.NS.nested_samples))],usemask=False)
        #self.posterior_samples = draw_posterior_many([self.nested_samples],[self.NS.Nlive],verbose=self.verbose)
        #np.savetxt(os.path.join(self.NS.output_folder,'posterior.dat'),self.posterior_samples.ravel(),header=' '.join(self.posterior_samples.dtype.names),newline='\n',delimiter=' ')
        #if self.verbose>1: self.plot()

class DynamicNestedSampler(NestedSampler):
    """
    Dynamic nested sampling algorithm
    From Higson, Handley, Hobson, Lazenby 2017
    https://arxiv.org/pdf/1704.03459.pdf
    """
    def __init__(self,usermodel, pipes=None, Ninit=10, output='.', verbose=1, seed=1, G=0.25, stopping=0.1):
        """
        G:      Goal parameter between 0 and 1.
                0: Optimise for evidence calculation
                1: Optimise for posterior sampling
        Ninit:  Initial number of live points
        """
        if pipes is None:
            self.pipes = []
        else:
            self.pipes = pipes
        self.model=usermodel
        self.G = G
        self.Ninit = Ninit
        self.setup_random_seed(seed)
        self.verbose = verbose
        self.accepted = 0
        self.rejected = 1
        self.queue_counter = 0
        self.Nlive=Ninit
        self.tolerance = stopping
        self.condition = np.inf
        self.logLmax = -np.inf
        self.iteration = 0
        self.nested_intervals=None
        self.state = _NSintegralState(self.Nlive)
        sys.stdout.flush()
        self.output_folder = output
        self.output,self.evidence_out,self.checkpoint = self.setup_output(output)
        header = open(os.path.join(output,'header.txt'),'w')
        header.write('\t'.join(self.model.names))
        header.write('\tlogL\n')
        header.close()
        self.logLmin = Value(c_double,-np.inf,lock=Lock())
        self.blocked=False
        self.reset()

    def connect_sampler(self,conn):
        """
        Connect to sampler on given connection
        """
        if self.verbose >1: print('Incoming connection from {0}'.format(str(conn)))
        #self.push_points(self.params,conn)
        #if self.verbose >1: print('Pushed {0} points to new connection'.format(len(self.params)))
        self.pipes.append(conn)

    def reset(self):
        self.nested_intervals=Interval(-np.inf, np.inf)
        
        samples = []
        
        for i in range(self.Ninit):
            tmp = self.model.new_point()
            tmp.logL = self.model.log_likelihood(tmp)
            samples.append(tmp)
        
        worst = np.argmin([p.logL for p in samples])
        self.nested_intervals.insert_point(samples[worst].logL, data=samples[worst])
        self.nested_intervals.find(-np.inf).n+=1
        for i in range(self.Ninit):
            if i!=worst:
                if self.verbose:
                    sys.stderr.write("Initial sampling --> {0:.0f} % complete\r".format((100.0*float(i+1)/float(self.Ninit))))
                    sys.stderr.flush()
                self.nested_intervals.insert_interval(Interval(-np.inf,samples[i].logL, data={samples[i].logL:samples[i]}))
        if self.verbose:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def terminate(self):
        """
        Returns True if termination condition met
        """
        pass

    @property
    def params(self):
        params=[]
        for p in list(self.nested_intervals.points()):
            params.append(self.nested_intervals.get_data(p))
        return params

    def push_points(self,points, conn=None):
        """
        Push points to connected samplers
        If conn is None, will push to all known
        """
        print(__name__,'Pushing ',points)
        if conn is None:
            conns = self.pipes
        else:
            conns = [conn]
        for c in conns:
            for p in points:
                try:
                    c.send(p)
                except (BrokenPipeError, ConnectionResetError):
                    print('Cannot push points to broken pipe')
                    self.blocked=True

    def recv_point(self):
        """
        Return a point from the next thread in the round-robin
        """

        point = None
        while len(self.pipes)>0:
            for r in mp.connection.wait(self.pipes):
                try:
                    data = r.recv()
                except (EOFError,ConnectionResetError,BrokenPipeError,StopIteration):
                    r.close()
                    self.pipes.remove(r)                    
                else:
                    if data is None:
                        r.close()
                        self.pipes.remove(r)
                    else:
                        yield data

    def nested_sampling_loop(self):
        """
        main nested sampling loop
        """
        # send all live points to the samplers for start
        #self.push_points(self.params)
        
        if self.verbose:
            sys.stderr.write("\n")
            sys.stderr.flush()
        
        # Run main loop
        logLmin=-np.inf
        while not self.done():
            self.evolve_classic(logLmin=logLmin)
            logLmin = self.params[self.iteration].logL
            self.iteration+=1
            

	    # Signal worker threads to exit
        self.logLmin.value = np.inf
        self.push_points([None])
        while True:
            try:
                x=next(self.recv_point())
                print(__name__,'Recived ',str(x))
            except StopIteration:
                break
            


    def evolve_classic(self, logLmin=-np.inf):
        """
        Perform an iteration of the classic nested sampling algorithm
        Draw a point above logLmin and add it to the set
        """
        # Find next point above logLmin
        i = self.nested_intervals.find(logLmin)
        oldpoint = self.nested_intervals.get_data(i.b)
        self.logLmin.value = np.float128(oldpoint.logL)        
        self.push_points([oldpoint])
        try:
            data = next(self.recv_point())
            print(__name__,' received ',str(data))
            if data is not None:
                self.acceptance,self.jumps, newpoint = data
                self.nested_intervals.insert_interval(Interval(oldpoint.logL, newpoint.logL, data={oldpoint.logL:oldpoint, newpoint.logL:newpoint}))
                if newpoint.logL > self.logLmax: self.logLmax = newpoint.logL
        except StopIteration:
            self.blocked=True
        
        logZ=self.nested_intervals.logZ()
        
        self.condition = logaddexp(logZ,self.logLmax - self.iteration/(float(self.Ninit))) - logZ


    def done(self):
        if self.blocked:
            return True
        if self.condition > self.tolerance:
            return False
        return True

    def run(self):
        """
        Run the algorithm
        """
        importance_frac = 0.9 # Minimum fraction of greatest importance to use in refinement
        termination = False
        while not self.terminate():
            # Recalculate importance function
            importance = self.importance()
            maxI_idx = np.argmax(importance)
            maxI = importance[maxI_idx]
            Lmax = self.logLs[maxI_idx]
            Lmin = min( self.logLs[importance>importance_frac*maxI] )
            self.sample_between(Lmin, Lmax)

class Interval(object):
    """
    Represents an interval
    """
    def __init__(self,a,b,n=0,parent=None,data=None):
        if b<a: a,b = b,a
        self.a=a
        self.b=b
        self.children=None
        self.parent=parent
        self.n=n
        self.data={}
        if data is not None:
            self.data=data
    
    def logt(self):
        if self.n==0:
            return 0.0
        else:
            return np.log(self.n-1)-np.log(self.n)
    
    def readout(self):
        """
        Returns logX,logL
        """
        logX=[0.0]
        logL=[-np.inf]
        for i in self:
            logX.append(logX[-1]+i.logt())
            logL.append(i.a)
        return (np.array(logX),np.array(logL))
    
    def logZ(self):
        """
        Return integral
        """
        logX,logL = self.readout()
        return log_integrate_log_trap(logL[1:-1], logX[1:-1])

    def get_data(self, key):
        """
        Get data from tree with given key
        """
        if key not in self:
            return None
        if self.children is not None:
            for c in self.children:
                if key in c:
                    return c.get_data(key)
        else:
            return self.data.get(key)
    
    def integrate(self,func):
        """
        Integrate the function over the interval
        """
        return np.sum(
            [ 0.5*(func(i.b)+func(i.a))*(i.logX()) for i in self]
            )
        
    def __contains__(self,x):
        if isinstance(x,Interval):
            return self.a <= x.a <= x.b <= self.b
        return self.a <= x <= self.b
    
    def __str__(self):
        return "({0}, {1}): n={2}".format(self.a,self.b,self.n)
    
    def leaves(self):
        if self.children is not None:
            yield self
        else:
            for c in self.children:
                yield c.leaves()
    
    def print_tree(self,pref=''):
        
        if self.children:
            print(pref+str(self))
            self.children[0].print_tree(pref=pref+u'|')
            self.children[1].print_tree(pref=pref+u'|')
        else:
            print(pref+str(self))

    def print_leaves(self,pref=''):
        if self.children:
            for c in self.children:
                c.print_leaves(pref=pref+'  ')
        else:
            print(pref+str(self))

    def __iter__(self):
        return next(self)
    
    def __next__(self):
        """
        Iterate over intervals
        """
        if self.children is not None:
            for c in self.children:
                for z in c:
                    yield z
        else:
            yield self

    def tree(self):
        """
        iterate over the tree
        """
        if self.children is not None:
            for c in self.children:
                for z in c.tree():
                    yield z
        else: yield self
        return

    #python 2 compatibility
    def next(self): return self.__next__()

    def find(self,x):
        """
        Return sub-interval containing x
        """
        if not x in self: return None
        if self.children is not None:
            for c in self.children:
                if x in c: return c.find(x)
        else:
            return self
    
    def split(self,x,data=None):
        """
        Split the interval at x, recording data if given
        """
        if x in self:
            ldata = {self.a:self.get_data(self.a), x:data}
            rdata = {x:data, self.b:self.get_data(self.b)}
            return [Interval(self.a,x,n=self.n,parent=self,data=ldata),Interval(x,self.b,n=self.n,parent=self,data=rdata)]
        else:
            return [self]
    
    def insert_point(self, x, data=None):
        """
        Insert a point into this interval tree
        """
        if self.a==x or self.b==x: return
        if not x in self:
            return
        if self.children is not None:
            for c in self.children:
                if x in c:
                    c.insert_point(x, data=data)
        else:
            self.children=self.split(x,data=data)

    def insert_interval(self,i):
        """
        Add an interval into the tree
        """
        # Subdivide left and right intervals with beginning and end of i
        self.insert_point(i.a, data=i.get_data(i.a))
        self.insert_point(i.b, data=i.get_data(i.b))
        # Find all intervals intersecting i and increase their multiplicity
        cur = self.find(i.a)
        for cur in self:
            if cur in i:
                cur.n+=1
    
    def __add__(self,other):
        if self.b == other.a: return Interval(self.a,other.b)
        elif self.a == other.b : return Interval(other.a,self.b)
        else: raise Exception("Cannot add non-contiguous intervals ({0},{1}), ({2},{3})".format(self.a,self.b,other.a,other.b))

    def points(self):
        #yield self.a
        for i in self:
            yield i.b

