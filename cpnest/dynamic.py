class DynamicNestedSampler(object):
    """
    Dynamic nested sampling algorithm
    From Higson, Handley, Hobson, Lazenby 2017
    https://arxiv.org/pdf/1704.03459.pdf
    """
    def __init__(self,G=0.25,Ninit=1000):
        """
        G:      Goal parameter between 0 and 1.
                0: Optimise for evidence calculation
                1: Optimise for posterior sampling
        Ninit:  Initial number of live points
        """
        self.G = G
        self.Ninit = Ninit
        pass
    def terminate(self):
        """
        Returns True if termination condition met
        """
    def importance(self):
        """
        Compute the importance function
        """
        pass
    def run(self):
        """
        Run the algorithm
        """
        importance_frac = 0.9 # Minimum fraction of greatest importance to use in refinement
        termination = False
        while not self.terminate():
            # Recalculate importance function
            importance = self.importance()
            maxI_idx = np.argmax(importance)
            maxI = importance[maxI_idx]
            Lmax = self.logLs[maxI_idx]
            Lmin = min( self.logLs[importance>importance_frac*maxI] )
            self.sample_between(Lmin, Lmax)
            
class DynamicNestState(object):
    """
    Stores the state of dynamic nest
    """
    def __init__(self):
        self.logL=[]
        self.n=[]
        
    @property
    def logt(self):
        """
        E[ log(t) ]
        """
        return -1/self.n

    @property
    def Var_logt(self):
        """
        Var [ log(t) ]
        """
        return 1/self.n**2
    
    def Z(self,t):
        """
        Evidence in dead points
        Z(t) = sum_{dead} L_i w_i(t)
        """
        pass
    
    def logX(self,logt):
        """
        Integral mass inside contour t
        """
        result = 0
        while(logt<result):
            result=np.logaddexp(result,next(self.logt))
        return result
    
    def w(self,t):
        """
        weight of sample at shrinkage t
        """
        pass
    
    
    def I_z(self):
        """
        Importance of current set to evidence
        I_z[i] = E[Z_{>i}] / n_i
        """
        
        
        
    

class DynamicNestState(object):
  """
  Stores the state of the nested sampling integrator
  """
  def __init__(self,nlive):
    self.nlive=nlive
    self.reset()
  def reset(self):
    """
    Reset the sampler
    """
    self.iteration=0
    self.logZ=-inf
    self.oldZ=-inf
    self.logw=0
    self.info=0
    # Start with a dummy sample enclosing the whole prior
    self.logLs=[-inf] # Likelihoods sampled
    self.log_vols=[0.0] # Volumes enclosed by contours
  def increment(self,logL,nlive=None):
    """
    Increment the state of the evidence integrator
    Simply uses rectangle rule for initial estimate
    """
    if(logL<=self.logLs[-1]):
      print('WARNING: NS integrator received non-monotonic logL. {0:.3f} -> {1:.3f}'.format(self.logLs[-1],logL))
    if nlive is None:
      nlive = self.nlive
    oldZ = self.logZ
    logt=-1.0/nlive
    Wt = self.logw + logL + logsubexp(0,logt)
    self.logZ = logaddexp(self.logZ,Wt)
    # Update information estimate
    if np.isfinite(oldZ) and np.isfinite(self.logZ):
      self.info = exp(Wt - self.logZ)*logL + exp(oldZ - self.logZ)*(self.info + oldZ) - self.logZ
    
    # Update history
    self.logw += logt
    self.iteration += 1
    self.logLs.append(logL)
    self.log_vols.append(self.logw)
  def finalise(self):
    """
    Compute the final evidence with more accurate integrator
    Call at end of sampling run to refine estimate
    """
    from scipy import integrate
    # Trapezoidal rule
    self.logZ=nest2pos.log_integrate_log_trap(np.array(self.logLs),np.array(self.log_vols))
    return self.logZ
