import logging

FMT = '%(asctime)s - %(name)-8s: %(message)s'
LEVELS = ['CRITICAL', 'WARNING', 'INFO', 'DEBUG']

def start_logger(output=None, verbose=0, name='CPNest'):
    """
    Start an instance of Logger for logging

    output : `str`
        output directory (./)

    verbose: `int`
        Verbosity, 0=CRITICAL, 1=WARNING, 2=INFO, 3=DEBUG

    name   : `str`
        Name of the logger (CPnest)
    """
    verbose = min(verbose, 3)
    # levels 0, 1, 2, 3
    level = LEVELS[verbose]
    # setup logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # handle command line output
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(FMT, datefmt='%Y-%m-%d, %H:%M:%S'))
    logger.addHandler(ch)

    # log to file
    if output is not None:
        add_file_handler(logger, output)

    logger.warning('Logging level: {}'.format(level))
    return logger

def add_file_handler(logger, output):
    """Add a file handler to a logger"""
    fh = logging.FileHandler(output + 'cpnest.log')
    fh.setFormatter(logging.Formatter(FMT, datefmt='%Y-%m-%d, %H:%M:%S'))
    logger.addHandler(fh)
    return logger

def update_logger(logger, verbose=None, output=None):
    """Update the verbosity and/or add an output file"""
    if output is not None:
        add_file_handler(logger, output)
    if verbose is not None:
        level = LEVELS[verbose]
        logger.setLevel(level)
        # update any handlers
        for handler in logger.handlers:
            handler.setLevel(level)
    return logger

