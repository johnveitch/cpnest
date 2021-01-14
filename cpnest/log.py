#! /usr/bin/env python
# coding: utf-8
import logging

# Default formats and level names
FMT = '%(asctime)s - %(name)-8s: %(message)s'
DATE_FMT = '%Y-%m-%d, %H:%M:%S'
LEVELS = ['CRITICAL', 'WARNING', 'INFO', 'DEBUG']

def file_handler(output, verbose=0):
    """
    Make a file handler with consistent formatting

    output : `str`
        Output directory
    """
    fh = logging.FileHandler(output + 'cpnest.log')
    fh.setFormatter(logging.Formatter(FMT, datefmt=DATE_FMT))
    return fh

def stream_handler():
    """
    Make a stream handler with consistent formatting
    """
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(FMT, datefmt=DATE_FMT))
    return sh
