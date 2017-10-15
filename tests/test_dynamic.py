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

def logLtest(x):
    return np.log(x)

xs = np.random.rand(10)

chain = [LivePoint(logL=logLtest(x),params=x) for x in xs]


print('logZ true: {0}'.format(np.log(0.5)))

New=Contour()
New.update(chain)
print('logZ contours: {0}'.format(New.logZ))

for iter in range(40):
    I=New.I()
    condition = I>0.9*max(I)
    Isel=np.where(condition)[0]
    Lsel=[New._logls[i] for i in Isel]
    logLmin=Lsel[0]
    logLthresh=Lsel[-1]
    samps=[]
    newls=[]
    print('searching between logP {0} and {1}'.format(Isel[0],Isel[-1]))
    while True:
        samps.append(np.random.uniform(np.exp(logLmin),1))
        newls.append(logLtest(samps[-1]))
        if max(newls)>logLthresh: break
    New.update([LivePoint(logL=logLtest(x),params=x) for x in samps])
    print('Inserted {0} points, {1} total, logZ: {2}'.format(len(samps),New.n,New.logZ))
    logX = list(x.logP for x in New.nested)
    plt.plot(logX, New.I(),label='iter {0}'.format(iter))
plt.legend()
plt.show()

#NS = cpnest.dynamic.DynamicNestState(live_points=chain)
#print('logZ {0}'.format(NS.logZ))

