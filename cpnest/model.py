from abc import ABCMeta,abstractmethod,abstractproperty
from numpy import inf
from .parameter import LivePoint
from numpy.random import uniform

class Model(object):
  """
  Base class for user's model. User should subclass this
  and implement log_likelihood, names and bounds
  """
  __metaclass__ = ABCMeta
  names=[] # Names of parameters, e.g. ['p1','p2']
  bounds=[] # Bounds of prior as list of tuples, e.g. [(min1,max1), (min2,max2), ...]
  def in_bounds(self,param):
    """
    Checks whether param lies within the bounds
    """
    return all(self.bounds[i][0] < param.values[i] < self.bounds[i][1] for i in range(param.dimension))
  
  def new_point(self):
    """
    Create a new LivePoint, drawn from within bounds
    """
    logP=-inf
    while(logP==-inf):
      p = LivePoint(self.names,[uniform(self.bounds[i][0],self.bounds[i][1]) for i,_ in enumerate(self.names)] )
      logP=self.log_prior(p)
    return p
  
  @abstractmethod
  def log_likelihood(self,param):
    """
    returns log likelihood of given parameter
    """
    pass
  def log_prior(self,param):
    """
    Returns log of prior.
    Default is flat prior within bounds
    """
    if self.in_bounds(param):
      return 0.0
    else: return -inf
    
  def strsample(self,sample):
    """
    Return a string representation for the sample to be written
    to the output file. User may overload for additional output
    """
    line='\t'.join('{0:.20e}'.format(sample[n]) for n in sample.names)
    line+='{0:20e}'.format(sample.logL)
    return line
  def header(self):
    """
    Return a string with the output file header
    """
    return '\t'.join(self.names) + '\tlogL'