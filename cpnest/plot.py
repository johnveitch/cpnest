import matplotlib as mpl
from matplotlib import pyplot as plt

def plot_chain(x,name=None,filename=None):
    """
    Produce a trace plot
    """
    fig=plt.figure(figsize=(4,3))
    plt.plot(x,',')
    plt.xlabel('iteration')
    if name is not None:
        plt.ylabel(name)
        if filename is None:
            filename=name+'_chain.png'
    if filename is not None:
        plt.savefig(filename)


def plot_hist(x,name=None,filename=None):
    """
    Produce a histogram
    """
    fig=plt.figure(figsize=(4,3))
    plt.hist(x,normed=True)
    plt.ylabel('density')
    if name is not None:
        plt.xlabel(name)
        if filename is None:
            filename=name+'_hist.png'
    if filename is not None:
        plt.savefig(filename)

def plot_corner(xs,filename=None,**kwargs):
    """
    Produce a corner plot
    """
    import corner
    fig=plt.figure(figsize=(10,10))
    corner.corner(xs,**kwargs)
    if filename is not None:
        plt.savefig(filename)

