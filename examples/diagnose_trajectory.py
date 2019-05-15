import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys, os

np.seterr(all='raise')

def log_likelihood(x):
    return np.sum([-0.5*x[n]**2-0.5*np.log(2.0*np.pi) for n in range(x.shape[0])])

mode = sys.argv[1]
if mode == 'delete':
    allfiles = os.listdir('.')
    toremove = [a for a in allfiles if 'trajectory_' in a and '.py' not in a]
    for f in toremove: os.remove(f)
    exit()
traj = np.genfromtxt('trajectory_'+sys.argv[1]+'.txt', names= True)
npts = 256
x = np.linspace(-10,10,npts)
y = np.linspace(-10,10,npts)
X, Y = np.meshgrid(x, y)
Z   = np.zeros((npts,npts))
for i in range(npts):
    for j in range(npts):
        Z[i,j] = log_likelihood(np.array([x[i],y[j]]))

C = plt.contour(X, Y, Z, levels = [traj['logLmin'][0]], linewidths=1.0,colors='k')
plt.contourf(X, Y, Z, 6, cmap = cm.Greys_r)
S = plt.scatter(traj['0'], traj['1'], c=traj['logL'], s = 8)
plt.plot(traj['0'],traj['1'], color = 'k', lw = 0.5)
for k in range(traj.shape[0]):
    plt.text(traj['0'][k],traj['1'][k], str(k), color="black", fontsize=8)
plt.colorbar(S)
plt.show()
