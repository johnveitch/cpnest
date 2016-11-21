import numpy as np

class gaussianmodel(object):
    def __init__(self):
        pass
    par_names=['mean2','sigma2']
    bounds=[[-10,10],[0.01,1]]
    data = np.array([x for x in np.random.normal(0.5,0.5,size=1000)])

    @classmethod
    def log_likelihood(cls,x):
        return -0.5*np.sum((cls.data-x.get('mean2'))**2/x.get('sigma2')**2) - len(cls.data)*np.log(x.get('sigma2')) - 0.5*np.log(2.0*np.pi)-1000
    @classmethod
    def log_prior(cls,p):
        for i in range(p.dimension):
            if not p.parameters[i].inbounds(): return -np.inf
        return 0



if __name__=='__main__':
    import cpnest

    work=cpnest.setup(gaussianmodel,verbose=1,Nthreads=8)
    work.run()


