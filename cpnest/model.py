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
        
        -----------
        Parameters:
            param: :obj:`cpnest.parameter.LivePoint`
        
        -----------
        Return:
            True: if all dimensions are within the bounds
            False: otherwise
        """
        return all(self.bounds[i][0] < param.values[i] < self.bounds[i][1] for i in range(param.dimension))
  
    def new_point(self):
        """
        Create a new LivePoint, drawn from within bounds
        
        -----------
        Return:
            p: :obj:`cpnest.parameter.LivePoint`
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
        
        ------------
        Parameter:
            param: :obj:`cpnest.parameter.LivePoint`
        """
        pass

    def log_prior(self,param):
        """
        Returns log of prior.
        Default is flat prior within bounds
        
        ----------
        Parameter:
            param: :obj:`cpnest.parameter.LivePoint`
        
        ----------
        Return:
            0 if param is in bounds
            -np.inf otherwise
        """
        if self.in_bounds(param):
            return 0.0
        else: return -inf

    def potential(self,param):
        """
        returns the potential energy as minus the log prior
        ----------
        Parameter:
        param: :obj:`cpnest.parameter.LivePoint`
        
        ----------
        Return:
            :obj: -`cpnest.model.log_prior`
        """
        return -self.log_prior(param)

    def force(self,param):
        """
        returns the force (-grad potential)
        Required for Hamiltonian sampling
        
        ----------
        Parameter:
        param: :obj:`cpnest.parameter.LivePoint`
        """
        pass

    def strsample(self,sample):
        """
        Return a string representation for the sample to be written
        to the output file. User may overload for additional output
        ----------
        Parameter:
            param: :obj:`cpnest.parameter.LivePoint`
        ----------
        Return:
            line: :string:
        """
        line='\t'.join('{0:.20e}'.format(sample[n]) for n in sample.names)
        line+='{0:20e}'.format(sample.logL)
        return line

    def header(self):
        """
        Return a string with the output file header
        """
        return '\t'.join(self.names) + '\tlogL'
