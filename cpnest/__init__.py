import logging
from .utils import stream_handler, LEVELS

# Configure base logger for the cpnest package - applies to loggers with names 
# prefixed by 'cpnest.'
_logger = logging.getLogger('cpnest')
_logger.addHandler(stream_handler())
_logger.setLevel(LEVELS[0])  # default verbosity of 0

from .cpnest import CPNest

# Get the version number from git tag
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = "dev"

__all__ = ['model',
           'NestedSampling',
           'parameter',
           'sampler',
           'cpnest',
           'nest2pos',
           'proposal',
           'plot',
           'logger']
