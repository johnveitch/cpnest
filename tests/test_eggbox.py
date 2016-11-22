import unittest
import numpy as np
import cpnest

class EggboxModel(object):
    """
    Eggbox problem from https://arxiv.org/pdf/0809.3437v1.pdf
    """
    def __init__(self):
        pass
    par_names=['x','y']
    bounds=[[0,10.0*np.pi],[0,10.0*np.pi]]
    data = None
    @classmethod
    def log_likelihood(cls,x):
        return log_eggbox(x['x'],x['y'])

    @staticmethod
    def log_prior(p):
        for i in range(p.dimension):
            if not p.parameters[i].inbounds(): return -np.inf
        return 0.0

def log_eggbox(x, y):
    tmp = 2.0+np.cos(x/2.)*np.cos(y/2.)
    return -5.0*np.log(tmp)

class EggboxTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(EggboxModel,verbose=1,Nthreads=1,Nlive=100,maxmcmc=100)

    def test_run(self):
        self.work.run()



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
