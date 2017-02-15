
from __future__ import division
from functools import reduce
import numpy as np
from numpy import log
from abc import ABCMeta,abstractmethod

def choice(a,size=None,replace=True,p=None):
    idx = np.random.choice(range(len(a)),size=size,replace=replace,p=p)
    # Mimic the behaviour of np.random.choice and return the element itself if size==1
    if type(idx)==np.int64:
        return a[idx]
    else:
        return [a[i] for i in idx]

class Proposal(object):
    """
    Base class for jump proposals
    """
    __metaclass__ = ABCMeta
    log_J = 0.0 # Jacobean of this jump proposal
    @abstractmethod
    def get_sample(self,old):
        """
        Returns a new proposed sample given the old one.
        """
        pass

class EnsembleProposal(Proposal):
    """
    Base class for ensemble proposals
    """
    ensemble=None
    def set_ensemble(self,ensemble):
        """
        Set the ensemble of points to use
        """
        self.ensemble=ensemble

class ProposalCycle(EnsembleProposal):
    """
    A proposal that cycles through a list of
    jumps.
    """
    idx=0 # index in the cycle
    N=0   # numer of proposals in the cycle
    def __init__(self,proposals,weights,cyclelength=100,*args,**kwargs):
        super(ProposalCycle,self).__init__(*args,**kwargs)
        # Normalise the weights
        norm = sum(weights)
        for i,_ in enumerate(weights):
            weights[i]=weights[i]/norm
        self.proposals=proposals
        # The cycle is a list of indices for self.proposals
        self.cycle = choice(self.proposals, size = cyclelength, p=weights, replace=True)
        self.N=len(self.cycle)

    def get_sample(self,old):
        # Call the current proposal and increment the index
        p = self.cycle[self.idx]
        new = p.get_sample(old)
        self.log_J = p.log_J
        self.idx = (self.idx + 1) % self.N
        return new

    def set_ensemble(self,ensemble):
        self.ensemble=ensemble
        for p in self.proposals:
            if isinstance(p,EnsembleProposal):
                p.set_ensemble(self.ensemble)

class EnsembleWalk(EnsembleProposal):
    log_J = 0.0 # Symmetric proposal
    def get_sample(self,old):
        Nsubset = 3
        subset = choice(self.ensemble,size=Nsubset,replace=False)
        center_of_mass = reduce(type(old).__add__,subset)/float(Nsubset)
        out = old.copy()
        for j in range(Nsubset):
            out += (subset[j] - center_of_mass)*np.random.normal(0,1)
        return out

class EnsembleStretch(EnsembleProposal):
    def get_sample(self,old):
        scale = 2.0 # Will stretch factor in (1/scale,scale)
        # Pick a random point to move toward
        a = choice(self.ensemble,size=1)[0]
        # Pick the scale factor
        x = np.random.uniform(-1,1)*log(scale)
        Z = np.exp(x)
        out = a + (old - a)*Z
        # Jacobean
        self.log_J = out.dimension * x
        return out

class DifferentialEvolution(EnsembleProposal):
    log_J = 0.0 # Symmetric jump
    def get_sample(self,old):
        a,b = choice(self.ensemble,size=2,replace=False)
        sigma = 1e-4 # scatter around difference vector by this factor
        out = old + (b-a)*np.random.normal(0,sigma)
        return out

class DefaultProposalCycle(ProposalCycle):
    def __init__(self,*args,**kwargs):
        proposals = [EnsembleWalk(), EnsembleStretch(), DifferentialEvolution()]
        weights = [1.0,1.0,1.0]
        super(DefaultProposalCycle,self).__init__(proposals,weights,*args,**kwargs)

