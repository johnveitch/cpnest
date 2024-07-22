import logging
from .utils import StreamHandler, LEVELS

# Configure base logger for the cpnest package - inherited by all loggers with
# names prefixed by 'cpnest'
logger = logging.getLogger('cpnest')
logger.setLevel(LEVELS[-1])  # maximum verbosity recorded to base logger
logger.addHandler(logging.NullHandler())

console_handler = StreamHandler(verbose=0)  # default console verbosity is 0
logger.addHandler(console_handler)
# To change the console handler verbosity:
#   from cpnest import console_handler
#   console_handler.set_verbosity(2)

from .cpnest import CPNest

# Get the version number from git tag
from importlib.metadata import version as distribution_version
try:
    __version__ = distribution_version(__name__)
except ModuleNotFoundError:
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
