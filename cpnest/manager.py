from ctypes import c_double, c_int
from abc import ABCMeta,abstractmethod
import multiprocessing as mp
from multiprocessing.managers import SyncManager,BaseManager
import numpy as np
import pickle

class JobCommand(object):
    pass

class CheckPointCommand(JobCommand):
    """
    This command tells a worker thread to checkpoint
    """

class UpdateLogLminCommand(JobCommand):
    """
    This command tells a sampler thread to update logLmin
    """
    def __init__(self,logLmin):
        self.logLmin=logLmin

class StopCommand(JobCommand):
    """
    Tell a worker thread to stop
    """
    pass

class StartCommand(JobCommand):
    """
    Tell a worker to start
    """
    pass

class TestCommand(JobCommand):
    """
    A command for testing
    """
    def __init__(self):
        print('Created test command {}'.format(self))

class ExitCommand(JobCommand):
    """
    Tell a worker thread to exit
    """
    def __init__(self, checkpoint=False):
        self.checkpoint=checkpoint

class CompletedSampleJob(object):
    def __init__(self, sample, acceptance=None, sub_acceptance=None,
                 jumps = None, proposed = None):
        self.sample=sample
        self.acceptance=acceptance
        self.sub_acceptance=sub_acceptance
        self.jumps=jumps
        self.proposed=proposed

class conns(object):
    connections=dict()
    producers=dict()
    def add_connection(self, thread, conn, prod):
        self.connections[thread]=conn
        self.producers[thread]=prod
    def get_connection(self, thread):
        return self.connections[thread]
    def send_all(self, x):
        for c in self.connections.values():
            c.send(x)
    def nclients(self):
        return len(self.connections)
    def get_all(self):
        return self.connections
    def connect_producer(self):
        """
        Returns the producer's end of the pipe
        """
        print('Connecting producer')
        #print('selfconnections',self.connections)
        con, prod = mp.Pipe()
        n = self.nclients()
        print('Connecting client {}'.format(n))
        con, prod = mp.Pipe()
        self.add_connection(n, con, prod)
        return prod, n

class RunManager(SyncManager):
    def __init__(self, address=None, authkey=None, maxclients=100, timeout=10):
        super(RunManager,self).__init__(address, authkey)
        self.timeout = timeout
        self.logLmin=mp.Value(c_double,-np.inf)
        self.connection_lock=mp.Lock()
        self.q_counter=0
        self.maxclients=maxclients
        self.checkpoint_flag=mp.Value(c_int,0)
        self.nclients=mp.Value(c_int,0)

    def start(self):
        super(RunManager, self).start()

        print('RunManager running at '+str(self.address))

    def connect_producer(self):
        return self.connections().connect_producer()
    
    def set_logLmin(self, logLmin):
        self.logLmin.value=logLmin
        
    def checkpoint_all(self):
        self.send_all(CheckPointCommand())

    def exit_all(self):
        self.send_all(ExitCommand())
            
    def send_all(self, command):
        self.connections().send_all(command)
            
    def start_all(self):
        print('Sending start command to {} producers'.format(self.connections().nclients()))
        self.send_all(StartCommand())

    def get_sample(self, fair_queue=False, block=True):
        """
        Return a sample from the output queue
        """
        while(True):
            try:
                if fair_queue:
                    conn = self.connections().get_connection(self.q_counter)[1]
                    if conn.poll(self.timeout):
                        s = conn.recv()
                        self.q_counter = (self.q_counter + 1) % self.nclients
                        return s
                    else: raise mp.TimeoutError
                else:
                    consumer_pipes=[c for c in self.connections().get_all().values()]
                    s = mp.connection.wait(consumer_pipes,
                                           timeout=self.timeout)
                    if s:
                        return s[0].recv()
                    else: raise mp.TimeoutError
            except mp.TimeoutError:
                print('Timeout waiting for new sample, your code is slow!')
                
class Worker(object):
    """
    Class for worker threads that knows how to communicate with
    the RunManager
    """
    def __init__(self, address=None, authkey=None, timeout=10):
        self.exit=False
        self.running=False
        self.conn=None
        self.threadid=None
        self.manager=None
        self.timeout=timeout
        self.address = address
        self.authkey = authkey
        self.connect(address=address, authkey=authkey)
        
    def connect(self, address=None, authkey=None):
        print('Connecting to manager at {}'.format(address))
        self.manager=RunManager(address, authkey)
        #self.manager.start()
        self.manager.connect() # called instead of start()
        #self.manager.start()
        print('connected logLmin, ',self.manager.logLmin)
        self.conn, self.threadid = self.manager.connections().connect_producer()
        print('Thread {} ({}) connected to manager at {}'.format(self.threadid,type(self),self.manager.address))
        print('Connection: ',id(self.conn))
        self.conn.send(TestCommand())
        
    def event_loop(self):
        """
        Main loop that processes job commands
        """
        print(type(self),'in event_loop')
        #print(self.conn.fileno())
        while True:
                #print('Polling self.conn {}'.format(type(self.conn)))
                polling = self.conn.poll(1)
                #print('Polled ',polling)
                if polling:
                    c = self.conn.recv()
                    print('Received {}'.format(c))
                    if c: self.process_command(c)
                else:
                    print('Sampler: No data read')
                    self.manager.start_all()
        print('Exiting event loop')
        return 0

    def process_command(self, command):
        """
        Process a command from `cpnest.manager`
        """
        print('Thread {} received {}'.format(self.threadid, command))
        if not isinstance(command, JobCommand):
            raise TypeError
        if isinstance(command, StartCommand):
            self.running=True
            self.do_work()
        if isinstance(command, CheckPointCommand):
            self.checkpoint()
        if isinstance(command, ExitCommand):
            if command.checkpoint:
                self.checkpoint()
            self.finish()
        if isinstance(command, TestCommand):
            print('TestCommand {} received by thread {} ({})'.format(command,
                                                                     self.threadid,
                                                                     type(self)
                                                                     )
                )
    
    def checkpoint(self, exit=False):
        """
        Checkpoint its internal state
        """
        with open(self.resume_file,"wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def resume(cls, resume_file, model, address, authkey):
        print('Resuming thread {} from '.format(self.threadid) + resume_file)
        with open(resume_file, "rb") as f:
            obj = pickle.load(f)
        obj.model   = model
        obj.connect(address, authkey) 
        return(obj)

#RunManager.register('producer_pipes')
#RunManager.register('consumer_pipes')
RunManager.register('connections',conns)
