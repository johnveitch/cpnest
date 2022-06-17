import numpy as np
import cpnest.model

def sine_gaussian(x,t):
    e = (t-x['t0'])/x['tau']
    return x['A']*np.exp(-e**2)*np.cos(2*np.pi*x['f']*t + x['phi'])

class BurstModel(cpnest.model.Model):
    """
    A Burst like search model
    """
    def __init__(self, time, data, sigma = 1.0):
        self.time   = time
        self.data   = data
        self.sigma  = sigma
        self.names  = ['A','f','tau','t0','phi']
        self.bounds = [[0.0,1.0],[10,100],[0.1,1.0],[time.min(),time.max()],[0.0,2*np.pi]]

    def log_likelihood(self,p):
        m = sine_gaussian(p, self.time)
        r = (self.data-m)/self.sigma
        return np.sum(-0.5*r**2)
    
    def log_prior(self,p):
        logP = super(BurstModel,self).log_prior(p)
        return logP
    
    def force(self, p):
        f = np.zeros(1, dtype = {'names':p.names, 'formats':['f8' for _ in p.names]})
        return f
    
    def analytical_gradient(self, p):
        return p.values

if __name__ == "__main__":
    np.random.seed(12)
    time  = np.linspace(0.0,1.0,1000)
    sigma = 0.1
    noise = np.random.normal(0,sigma,size = time.shape[0])
    truth = {'A':0.1,'f':50,'tau':0.05,'t0':0.5,'phi':1.0}
    signal = sine_gaussian(truth,time)
    data = noise+signal
    model=BurstModel(time, data, sigma=sigma)
    work=cpnest.CPNest(model, verbose=2, nnest=4, nensemble=8, nlive=1000, maxmcmc=2000, nslice=0, nhamiltonian=0, resume=1, periodic_checkpoint_interval=100, output='burst_test')
    work.run()
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(time,signal,lw=0.75,color='r',zorder=100)
    ax.plot(time,data,lw=0.5,color='k')
    pos = work.posterior_samples
    models = np.array([sine_gaussian(p,time) for p in pos])
    l,m,h  = np.percentile(models,[5,50,95],axis=0)
    ax.fill_between(time,l,h,facecolor='turquoise',alpha=0.5)
    ax.plot(time,m,lw=0.75,color='turquoise')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('strain')
    plt.savefig('burst_test/burst_waveform.pdf',bbox_inches='tight')
