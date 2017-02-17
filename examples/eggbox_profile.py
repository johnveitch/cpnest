import unittest
import numpy as np
import cpnest.model

"""
Once run this example, use

pyprof2calltree -i <profile report file> -k

for detail call tree information
"""

from eggbox import EggboxModel

class EggboxTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(EggboxModel(),verbose=2,Nthreads=8,Nlive=5000,maxmcmc=5000)

    def test_run(self):
        self.work.profile()

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    work=cpnest.CPNest(EggboxModel(),verbose=2,Nthreads=4,Nlive=1000,maxmcmc=5000)
    work.profile()
    #unittest.main(verbosity=2)
