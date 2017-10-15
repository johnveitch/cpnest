import unittest
import numpy as np
from scipy import integrate,stats
import cpnest
import cpnest.model
import cpnest.dynamic
from cpnest.dynamic import Contour,LivePoint
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
plt.ion()
import sys
import os
import time

def logLtest(x):
    return np.log(x)

xs = np.random.rand(10)

chain = [LivePoint(logL=logLtest(x),params=x) for x in xs]



New=Contour()
#New.update(chain)
New.add(chain[1])
print('logZ contours: {0}'.format(New.logZ))

for iter in range(100):
    I=New.I(G=0.25)
    condition = I>0.9*max(I)
    Isel=np.where(condition)[0]
    Lsel=[New._logls[i] for i in Isel]
    logLmin=Lsel[0]
    logLthresh=Lsel[-1]
    samps=[]
    newls=[]
    logX = list(x.logP for x in New.nested)
    plt.figure(1)
    plt.plot(logX, New._logls,label='iter {0}'.format(iter))
    #plt.legend()
    plt.figure(2)
    plt.semilogy(logX, I, label='iter {0}'.format(iter))
    plt.show(block=False)
    plt.pause(0.05)
    print('searching between logP {0} and {1}'.format(Isel[0],Isel[-1]))
    while True:
        samps.append(np.random.uniform(np.exp(logLmin),1))
        newls.append(logLtest(samps[-1]))
        if max(newls)>logLthresh: break
    New.update([LivePoint(logL=logLtest(x),logLmin=logLmin,params=x) for x in samps])
    print('Inserted {0} points, {1} total, logZ: {2}\n\n'.format(len(samps),New.n,New.logZ))
    
    #time.sleep(0.1)

#NS = cpnest.dynamic.DynamicNestState(live_points=chain)
#print('logZ {0}'.format(NS.logZ))

print('logZ true: {0}'.format(np.log(0.5)))
