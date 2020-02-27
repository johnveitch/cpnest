import unittest
import numpy as np
import jax.numpy as jnp
import jax
import cpnest.model

class EggboxModel(cpnest.model.Model):
    """
    Eggbox problem from https://arxiv.org/pdf/0809.3437v1.pdf
    """
    names=['0','1','2','3','4']
    bounds=[[0,10.0*np.pi],[0,10.0*np.pi],[0,10.0*np.pi],[0,10.0*np.pi],[0,10.0*np.pi]]
    data = None

    def log_likelihood(self,x):
        return log_eggbox(x, dim = len(x))

    def force(self,x):
        f = np.zeros(1, dtype = {'names':x.names, 'formats':['f8' for _ in x.names]})
        return f

def log_eggbox(p, dim = 1):
    tmp = 1.0
    for i in range(dim):
        tmp *= jnp.cos(p['{0:d}'.format(i)]/2.)
    return (tmp+2.0)**5.0

class EggboxTestCase(unittest.TestCase):
    """
    Test the eggox model
    """
    def setUp(self):
        self.work=cpnest.CPNest(EggboxModel(),verbose=1,nthreads=1,nlive=1000,maxmcmc=1000)

    def test_run(self):
        self.work.run()

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
        work=cpnest.CPNest(EggboxModel(),verbose=3,nthreads=4,nlive=1000,maxmcmc=1000,poolsize=1000, nhamiltonian=4)
        work.run()

