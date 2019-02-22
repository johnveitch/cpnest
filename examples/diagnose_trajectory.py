import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

np.seterr(all='raise')

def log_likelihood(x):
    return np.sum([-0.5*x[n]**2-0.5*np.log(2.0*np.pi) for n in range(x.shape[0])])

#plt.hist(np.exp(np.random.uniform(np.log(1),np.log(1000),size=10000)),bins=1000)
#plt.show()

traj = []
while traj == []:
    try:
        traj = np.loadtxt('trajectory.txt')
        break
    except:
        pass

npts = 256
x = np.linspace(-10,10,npts)
y = np.linspace(-10,10,npts)
X, Y = np.meshgrid(x, y)
Z   = np.zeros((npts,npts))
for i in range(npts):
    for j in range(npts):
        Z[i,j] = log_likelihood(np.array([x[i],y[j]]))

C = plt.contour(X, Y, Z, levels = [traj[0,-1]], linewidths=1.0,colors='k')
plt.contourf(X, Y, Z,6,cmap=cm.Greys_r)
S = plt.scatter(traj[:,0], traj[:,1], c=traj[:,-2], s = 8)
plt.plot(traj[:,0],traj[:,1], color = 'k', lw = 0.5)
for k in range(traj.shape[0]):
    print(k,traj[k,0],traj[k,-2],traj[0,-1])
    plt.text(traj[k,0],traj[k,1], str(k), color="black", fontsize=8)
plt.colorbar(S)
plt.show()
