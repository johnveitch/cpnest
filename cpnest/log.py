import os
import logging

# Default formats and level names
FMT = '%(asctime)s - %(name)-8s: %(message)s'
DATE_FMT = '%Y-%m-%d, %H:%M:%S'
LEVELS = ['CRITICAL', 'WARNING', 'INFO', 'DEBUG']

def file_handler(fname, path=''):
    """
    Make a file handler with consistent formatting

    fname : str
        Filename of log file

    path : str, optional
        Path to save the file. Defaults to current working directory
    
    """
    fh = logging.FileHandler(os.path.join(path, fname))
    fh.setFormatter(logging.Formatter(FMT, datefmt=DATE_FMT))
    return fh

def stream_handler():
    """
    Make a stream handler with consistent formatting
    """
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(FMT, datefmt=DATE_FMT))
    return sh
