import numpy as np
from scipy.interpolate import Rbf

class numerical_gradient:
    """
    numerical gradient class. It computes
    the gradient of a scalar function with respect to its
    arguments using first order numerical differentiation.
    The class stores its samples and attempts to
    interpolate for speed.
    Can be signalled to reinitialise
    """
    def __init__(self, dimension, function, cache_size = 1000):
        self.dimension    = dimension
        self.interpolant  = [None]*self.dimension
        self.step         = 1e-7
        self.function     = function
        self.index        = self.counter()
        self.k            = 0
        self.cache_size   = cache_size
        self.cache        = np.zeros((self.cache_size,self.dimension), dtype = np.float64)
        self.cache_values = np.zeros((self.cache_size,self.dimension), dtype = np.float64)
        self.brute_force  = True
    
    def __call__(self, x):
        """
        x has to be a live point
        """
        if self.brute_force == True:
            return self.finite_differencing(x)
        else:
            return self.interpolated_differencing(x)
    
    def counter(self):
        n = 0
        while True:
            yield n%self.cache_size
            n += 1
            
    def finite_differencing(self, x):
        """
        compute the central difference numerical derivative
        
        x has to be a np.ndarray
        """
        v  = np.zeros(self.dimension)
        xl = x.copy()
        xu = x.copy()
        for i,n in enumerate(x.names):
            xl[n] -= self.step
            xu[n] += self.step
            v[i]   = (self.function(xu)-self.function(xl))/(2*self.step)
            xl[n] += self.step
            xu[n] -= self.step
        
        oldk   = self.k
        self.k = next(self.index)
        self.cache[self.k, :]        = x.values
        self.cache_values[self.k, :] = v
#        if oldk == self.cache_size-1 and self.k == 0 and self.brute_force == True:
##            for j in range(self.cache_size): print(self.cache[j, :])
#            print(oldk,self.cache_size-1,self.k)
#            print('updated gradient state')
#            self.update_state()
#            exit()
        return v
    
    def interpolated_differencing(self, x):
        y = x.values
        v  = np.zeros(self.dimension)
        return np.array([self.interpolant[i](*np.transpose(y)) for i in range(self.dimension)])
    
    def interpolate(self):

        for i in range(self.dimension):
            self.interpolant[i] = Rbf(*self.cache.T, self.cache_values[:,i])
        return self.interpolant
    
    def update_state(self):
        if self.brute_force == True:
            self.interpolate()
            self.brute_force = False
        elif self.brute_force == False:
            self.brute_force = True

if __name__=="__main__":

    dim = 20
    def function(x):
        return np.sum([x[i]**2 for i in range(x.shape[0])])
        
        
    X = np.random.uniform(-10,10.,size=(3000,dim))

    N = numerical_gradient(dim, function)
    
    for i in range(X.shape[0]):
        if i == 1000 or i == 2000:
            print('switching computation mode')
            N.update_state()
        print("x = ",X[i], "num grad = ",N(X[i]))


