import unittest
import numpy as np
import cpnest.model

"""
Once run this example, use

pyprof2calltree -i <profile report file> -k

for detail call tree information
"""

from eggbox import EggboxModel

if __name__=='__main__':
    work=cpnest.CPNest(EggboxModel(),verbose=2,nthreads=4,nlive=1000,maxmcmc=5000)
    work.profile()
    #unittest.main(verbosity=2)
