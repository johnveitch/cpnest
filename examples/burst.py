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
    truth = {'A':0.1,'f':50,'tau':0.5,'t0':0.5,'phi':1.0}
    signal = sine_gaussian(truth,time)
    data = noise+signal
    model=BurstModel(time, data, sigma=sigma)
    work=cpnest.CPNest(model, verbose=2, nnest=4, nensemble=8, nlive=1024, maxmcmc=5000, nslice=0, nhamiltonian=0, resume=0)
    work.run()
