import logging
from .logger import CPNestLogger
from .cpnest import CPNest

# Get the version number from git tag
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

logging.setLoggerClass(CPNestLogger)

__all__ = ['model',
           'NestedSampling',
           'parameter',
           'sampler',
           'cpnest',
           'nest2pos',
           'proposal',
           'plot',
           'logger']

