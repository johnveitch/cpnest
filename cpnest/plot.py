import matplotlib as mpl
from matplotlib import pyplot as plt

def plot_chain(x,name=None,filename=None):
    """
    Produce a trace plot
    """
    fig=plt.figure(figsize=(4,3))
    plt.plot(x,',')
    plt.grid()
    plt.xlabel('iteration')
    if name is not None:
        plt.ylabel(name)
        if filename is None:
            filename=name+'_chain.png'
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close()

def plot_hist(x,name=None,filename=None):
    """
    Produce a histogram
    """
    fig=plt.figure(figsize=(4,3))
    plt.hist(x, density = True, facecolor = '0.5', bins=int(len(x)/20))
    plt.ylabel('probability density')
    if name is not None:
        plt.xlabel(name)
        if filename is None:
            filename=name+'_hist.png'
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close()

def plot_corner(xs,filename=None,**kwargs):
    """
    Produce a corner plot
    """
    import corner
    fig=plt.figure(figsize=(10,10))
    mask = [i for i in range(xs.shape[-1]) if not all(xs[:,i]==xs[0,i]) ]
    corner.corner(xs[:,mask],**kwargs)
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close()

