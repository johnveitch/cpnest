from abc import ABCMeta,abstractmethod


class JobCommand(ABCMeta):
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

class StartCommand(JobCommand):
    """
    Tell a worker to start
    """

class ExitCommand(JobCommand):
    """
    Tell a worker thread to exit
    """
    def __init__(self, checkpoint=False):
        self.checkpoint=checkpoint

class CompletedSampleJob(object):
    def __init__(self, sample, acceptance=None, sub_acceptance=None,
                 jumps = None, proposed = None):
        self.sample=samples
        self.acceptance=acceptance
        self.sub_acceptance=sub_acceptance
        self.jumps=jumps
        self.proposed=proposed

class RunManager(SyncManager):
    def __init__(self, timeout=60, **kwargs):
        super(RunManager,self).__init__(**kwargs)
        self.nconnected = mp.Value(c_int,0)
        self.timeout = timeout
        self.producer_pipes = self.list()
        self.consumer_pipes = self.list()
        self.q_counter=0
        for i in range(nthreads):
            consumer, producer = mp.Pipe(duplex=True)
            self.producer_pipes.append(producer)
            self.consumer_pipes.append(consumer)
        self.logLmin=None
        self.nthreads=nthreads
        self.output_queue = self.Queue()
        print('RunManager running at '+str(self.address))

    def start(self):
        super(RunManager, self).start()
        self.logLmin = mp.Value(c_double,-np.inf)
        self.checkpoint_flag=mp.Value(c_int,0)

    def connect_producer(self):
        """
        Returns the producer's end of the pipe
        """
        with self.nconnected.get_lock():
            n = self.nconnected.value
            pipe = self.producer_pipes[n]
            self.nconnected.value+=1
        return pipe, n
    
    def update_logLmin(self, logLmin):
        self.logLmin.value=logLmin
        
    def checkpoint_all(self):
        for p in self.producer_pipes:
            p.send(CheckPointCommand())

    def exit_all(self):
        for p in self.producer_pipes:
            p.send(ExitCommand())

    def get_sample(self, fair_queue=True, block=True, timeout=60):
        """
        Return a sample from the output queue
        """
        while(True):
            try:
                if fair_queue:
                    self.consumer_pipes[self.q_counter].poll(self.timeout)
                    s = self.consumer_pipes[self.q_counter].recv()
                    self.q_counter = (self.q_counter + 1) % len(self.consumer_pipes)
                else:
                    s = mp.connection.wait(self.consumer_pipes,
                                           timeout=self.timeout)
                return s
            except mp.TimeoutError:
                print('Timeout waiting for new sample, your code is slow!')
                
class Worker(object):
    """
    Class for worker threads that knows how to communicate with
    the RunManager
    """
    def __init__(self, address=None, authkey=None):
        self.manager=RunManager(address, authkey)
        self.exit=False
        self.running=False
        
    def event_loop(self):
        """
        Main loop that processes job commands
        """
        self.manager.connect()
        self.running=False

        while not self.exit:
            c = self.producer_pipe.recv()
            if c: self.process_command(c)
        return 0

    def process_command(self, command):
        """
        Process a command from `cpnest.manager`
        """
        if not isinstance(command, manager.JobCommand):
            raise TypeError
        if isinstance(command, manager.StartCommand):
            self.running=True
            self.do_work()
        if isinstance(command, manager.CheckPointCommand):
            self.checkpoint()
        if isinstance(command, manager.ExitCommand):
            if command.checkpoint:
                self.checkpoint()
            self.finish()
