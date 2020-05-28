import unittest
import numpy as np
import cpnest.model
from numpy import log

class AckleyModel(cpnest.model.Model):
    """
    Ackley problem from https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    names=['x','y']
    bounds=[[-5,5],[-5,5]]
    data = None
    @staticmethod
    def log_likelihood(x):
        return log(ackley(x['x'],x['y']))

    def force(self,x):
        f = np.zeros(1, dtype = {'names':x.names, 'formats':['f8' for _ in x.names]})
        return f

def ackley(x, y):
    """
    Ackley problem
    """
    r = np.sqrt(0.5*(x*x+y*y))
    first = 20.0*np.exp(r)
    second = np.exp(0.5*(np.cos(2.0*np.pi*x)+np.cos(2.0*np.pi*y)))
    return -(first+second-np.exp(1)-20)

class AckleyTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(AckleyModel(),verbose=1,nthreads=8,nlive=1000,maxmcmc=1000)

    def test_run(self):
        self.work.run()



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
