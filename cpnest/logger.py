import logging

class CPNestLogger(logging.Logger):
    """
    Custom logger class that inherits from the Logger class and is called when
    instantiating a logger with the logging package in CPNest.

    It includes a stream handler by default and can be updated to change
    verbosity and/or log to a file.
    ---------

    Initialisation arguments:

    args:

    name:
        :str: name of the logger
    """

    def __init__(self, name):

        super(CPNestLogger, self).__init__(name)

        self.fmt = '%(asctime)s - %(name)-8s: %(message)s'
        self.date_fmt = '%Y-%m-%d, %H:%M:%S'
        self.levels = ['CRITICAL', 'WARNING', 'INFO', 'DEBUG']
        self.add_stream_handler()

    def add_file_handler(self, output):
        """
        Add a file handler

        output : `str`
            Output directory
        """
        fh = logging.FileHandler(output + 'cpnest.log')
        fh.setFormatter(logging.Formatter(self.fmt, datefmt=self.date_fmt))
        self.addHandler(fh)

    def add_stream_handler(self):
        """
        Add a stream handler
        """
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(self.fmt, datefmt=self.date_fmt))
        self.addHandler(sh)

    def update(self, output=None, verbose=0):
        """
        Update the verbosity and/or add an output file

        verbose: `int`
            Verbosity, 0=CRITICAL, 1=WARNING, 2=INFO, 3=DEBUG

        output : `str`
            Output directory
        """
        verbose = min(verbose, 3)
        if output is not None:
            self.add_file_handler(output)
        if verbose is not None:
            level = self.levels[verbose]
            self.setLevel(level)
            # update any handlers
            for handler in self.handlers:
                handler.setLevel(level)
        self.warning('Setting verbosity to {}'.format(verbose))
