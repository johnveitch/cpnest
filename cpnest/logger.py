import logging

def start_logger(output=None, verbose=0):
    """
    Start an instance of Logger for logging

    output : `str`
        output directory (./)

    verbose: `int`
        Verbosity, 0=CRITICAL, 1=WARNING, 2=INFO, 3=DEBUG

    fmt: `str`
        format for logger (None) See logging documentation for details

    """
    # possible levels
    verbose = min(verbose, 3)
    # levels 0, 1, 2, 3
    levels = ['CRITICAL', 'WARNING', 'INFO', 'DEBUG']
    level = levels[verbose]
    fmt = '%(asctime)s - %(name)-8s: %(message)s'
    # setup logger
    logger = logging.getLogger('CPNest')
    logger.setLevel(level)
    # handle command line output
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%d, %H:%M:%S'))
    logger.addHandler(ch)

    if output is not None:
        # log to file
        fh = logging.FileHandler(output + 'cpnest.log')
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    print(logger.critical('Logging level: {}'.format(level)))
    return logger
