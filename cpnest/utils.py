import os
import logging

# Default formats and level names
FORMATTER = logging.Formatter('%(asctime)s - %(name)-8s: %(message)s', datefmt='%Y-%m-%d, %H:%M:%S')
LEVELS = ['CRITICAL', 'WARNING', 'INFO', 'DEBUG']
LOGGER = logging.getLogger('cpnest.utils')


class Handler(logging.Handler):

    def __init__(self, verbose=0, **kwargs):
        super().__init__(**kwargs)
        self.set_verbosity(verbose)
        self.setFormatter(FORMATTER)
    
    def get_verbosity(self):
        return self._verbose

    def set_verbosity(self, verbose):
        self._verbose = verbose
        self.setLevel(LEVELS[verbose])
        

class ConsoleHandler(Handler, logging.StreamHandler):
    def __init__(self, verbose=0, **kwargs):
        super().__init__(verbose=verbose, **kwargs)


class FileHandler(Handler, logging.FileHandler):
    def __init__(self, filename, verbose=0, **kwargs):
        super().__init__(filename=filename, verbose=verbose, **kwargs)


class LogFile:
    """
    Context manager for file logging. It logs everything from `logger` in some file at a given `filename`.

    Parameters
    ----------
    filename : str
        Filename under which to save the log.
    
    verbose : int, optional
        Logging level verbosity 0='CRITICAL' 1='WARNING' 2='INFO' 3='DEBUG'.
    
    loggername : str, optional
        Name of the logger to send to file at `path`. Default is `'cpnest'` so 
        all cpnest logs are recorded. E.g. specify `'cpnest.cpnest'` to only
        record logs from the `cpnest.py` module.

    Attributes
    ----------
    handler : logging.FileHandler
        File handler object.

    Examples
    --------
    ```python
    from cpnest.utils import LogFile

    with LogFile('example.log') as flog:
        # Do some stuff here and it will be logged to 'example.log'
        ...

    # Do some stuff here and it won't be logged to 'example.log'

    with flog:
        # Do some stuff here and it will be logged to 'example.log'
        ... 
    ```

    """

    def __init__(self, filename, verbose=0, loggername='cpnest'):
        self.filename = filename
        self.verbose = verbose
        self.logger = logging.getLogger(loggername)
        self.handler = None

    def __enter__(self):
        self.handler = FileHandler(self.filename, verbose=self.verbose)
        self.logger.addHandler(self.handler)
        return self
    
    def __exit__(self, type, value, traceback):
        self.logger.removeHandler(self.handler)
        self.handler.close()  # Safely closes file handler
        self.handler = None
