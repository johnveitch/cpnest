import logging
from .logger import CPNestLogger
from .cpnest import CPNest

logging.setLoggerClass(CPNestLogger)

__version__ = '0.9.9'

__all__ = ['model',
           'NestedSampling',
           'parameter',
           'sampler',
           'cpnest',
           'nest2pos',
           'proposal',
           'plot',
           'logger']

