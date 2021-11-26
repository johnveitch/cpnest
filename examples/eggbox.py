import unittest
import numpy as np
import cpnest.model

class EggboxModel(cpnest.model.Model):
    """
    Eggbox problem from https://arxiv.org/pdf/0809.3437v1.pdf
    """
    names=['1','2','3','4','5']
    bounds=[[0,10.0*np.pi],[0,10.0*np.pi],[0,10.0*np.pi],[0,10.0*np.pi],[0,10.0*np.pi]]
    data = None
    
    def log_likelihood(self,x):
        return log_eggbox(x)
    
    def log_prior(self,x):
        return super(EggboxModel,self).log_prior(x)

def log_eggbox(p):
    tmp = 1.0
    for n in p.names:
        tmp *= np.cos(p[n]/2.)
    return (tmp+2.0)**5.0

if __name__=='__main__':
    M = EggboxModel()
    work=cpnest.CPNest(M,verbose=2,nnest=4,nensemble=4,nlive=1000,maxmcmc=1000,nslice=4)
    work.run()

