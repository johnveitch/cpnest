import matplotlib as mpl
from matplotlib import pyplot as plt

def init_plotting():
    plt.rcParams['figure.figsize'] = (3.4, 3.4)
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

init_plotting()

def plot_chain(x, name=None, filename=None):
    """
    Produce a trace plot
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(x,',')
    ax.grid()
    ax.set_xlabel('iteration number')
    if name is not None:
        ax.set_ylabel(name)
        if filename is None:
            filename=name+'_chain.pdf'
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_hist(x, name=None, prior_samples=None, mcmc_samples=None, filename=None):
    """
    Produce a histogram
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.hist(x, density = True, color='black', linewidth = 1.25,
                histtype='step', bins=len(x)//50, label = 'posterior')
    if prior_samples is not None:
        ax.hist(prior_samples, density = True, color='green', linewidth = 1.0,
                histtype='step', bins=len(x)//50, label = 'prior')
    if mcmc_samples is not None:
        ax.hist(mcmc_samples, density = True, color='red', linewidth = 1.0,
                histtype='step', bins=len(x)//50, label = 'mcmc')
    ax.legend(loc='upper left')
    ax.set_ylabel('probability density')
    if name is not None:
        ax.set_xlabel(name)
        if filename is None:
            filename=name+'_hist.pdf'
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_indices(indices, filename=None, max_bins=30):
    """
    Histogram indices for insertion indices tests.

    Parameters
    ----------
    indices : list
        List of insertion indices
    filename : str, optional
        Filename used to saved resulting figure. If not specified figure
        is not saved.
    max_bins : int, optional
        Maximum number of bins in the histogram.
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    ax.hist(indices, density=True, color='tab:blue', linewidth=1.25,
                histtype='step', bins=min(len(indices) // 100, max_bins))
    # Theoretical distribution
    ax.axhline(1, color='black', linewidth=1.25, linestyle=':', label='pdf')

    ax.legend()
    ax.set_xlabel('Insertion indices [0, 1]')

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_corner(xs, ps=None, ms=None, filename=None, **kwargs):
    """
    Produce a corner plot
    """
    import corner
    mask = [i for i in range(xs.shape[-1]) if not all(xs[:,i]==xs[0,i]) ]
    fig  = corner.corner(xs[:,mask], color='k', hist_kwargs={'density':True}, **kwargs)
    if ps is not None:
        mask = [i for i in range(ps.shape[-1]) if not all(ps[:,i]==ps[0,i]) ]
        corner.corner(ps[:,mask], fig = fig, color='g', hist_kwargs={'density':True}, **kwargs)
    if ms is not None:
        mask = [i for i in range(ms.shape[-1]) if not all(ms[:,i]==ms[0,i]) ]
        corner.corner(ms[:,mask], fig = fig, color='r', hist_kwargs={'density':True}, **kwargs)
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close()

