import logging
import numpy as np
from numpy import logaddexp, vstack
from numpy.random import uniform
from functools import reduce
from scipy.signal import correlate
# if not logging.Logger.manager.loggerDict:
#     LOGGER = logging.getLogger('cpnest.nest2pos')
# else:
#     LOGGER = logging.getLogger('cpnest.cpnest.CPNest')

# Note - logger should take the name of the module. It inherits from the base
# cpnest logger.
LOGGER = logging.getLogger('cpnest.nest2pos')

def logsubexp(x,y):
    """
    Helper function to compute the exponential
    of a difference between two numbers

    ----------
    Parameter:
        x: :float:
        y: :float:
    ----------
    Return
        z: :float: x + np.log1p(-np.exp(y-x))
    """
    assert np.all(x >= y), 'cannot take log of negative number {0!s} - {1!s}'.format(str(x), str(y))
    return x + np.log1p(-np.exp(y-x))

def log_integrate_log_trap(log_func,log_support):
    """
    Trapezoidal integration of given log(func)
    Returns log of the integral
    """

    log_func_sum = logaddexp(log_func[:-1], log_func[1:]) - np.log(2)
    log_dxs = logsubexp(log_support[:-1], log_support[1:])

    return logaddexp.reduce(log_func_sum + log_dxs)


def compute_weights(data, Nlive):
    """Returns log_ev, log_wts for the log-likelihood samples in data,
    assumed to be a result of nested sampling with Nlive live points."""

    start_data=np.concatenate(([float('-inf')], data[:-Nlive]))
    end_data=data[-Nlive:]

    log_wts=np.zeros(data.shape[0])

    log_vols_start=np.cumsum(np.ones(len(start_data)+1)*np.log1p(-1./Nlive))-np.log1p(-1./Nlive)
    log_vols_end=np.zeros(len(end_data))
    log_vols_end[-1]=np.NINF
    log_vols_end[0]=log_vols_start[-1]+np.log1p(-1.0/Nlive)
    for i in range(len(end_data)-1):
        log_vols_end[i+1]=log_vols_end[i]+np.log1p(-1.0/(Nlive-i))

    log_likes = np.concatenate((start_data,end_data,[end_data[-1]]))

    log_vols=np.concatenate((log_vols_start,log_vols_end))
    log_ev = log_integrate_log_trap(log_likes, log_vols)

    log_dXs = logsubexp(log_vols[:-1], log_vols[1:])
    log_wts = log_likes[1:-1] + log_dXs[:-1]

    log_wts -= log_ev

    return log_ev, log_wts

def draw_posterior(data, log_wts, verbose=False, rng=None):
    """Draw points from the given data (of shape (Nsamples, Ndim))
    with associated log(weight) (of shape (Nsamples,)). Draws uniquely so
    there are no repeated samples"""
    if rng is None:
        rng = uniform

    maxWt=max(log_wts)
    normalised_wts=log_wts-maxWt
    selection=[n > np.log(rng()) for n in normalised_wts]
    idx=list(filter(lambda i: selection[i], range(len(selection))))
    return data[idx]

def draw_posterior_many(datas, Nlives, verbose=False, rng=None):
    """Draw samples from the posteriors represented by the
    (Nruns, Nsamples, Nparams)-shaped array datas, each sampled with
    the corresponding Nlive number of live points. Will draw without repetition,
    and weight according to the evidence in each input run"""
    
    if rng is None:
        generator = uniform
    else:
        generator = rng.random
    
    # list of log_evidences, log_weights
    
    log_evs,log_wts=zip(*[compute_weights(data['logL'],Nlive) for data,Nlive in zip(datas, Nlives)])
    if verbose:
        LOGGER.critical('Computed log_evidences: {0!s}'.format((str(log_evs))))

    log_total_evidence=reduce(logaddexp, log_evs)
    log_max_evidence=max(log_evs)
    #print 'evidences: %s'%(str(log_evs))
    fracs=[np.exp(log_ev-log_max_evidence) for log_ev in log_evs]
    if verbose:
        LOGGER.critical('Relative weights of input files: {0!s}'.format((str(fracs))))
    Ns=[fracs[i]/len(datas[i]) for i in range(len(fracs))]
    Ntot=max(Ns)
    fracs=[n/Ntot for n in Ns]
    if verbose:
        LOGGER.critical('Relative weights of input files taking into account their length: {0!s}'.format((str(fracs))))

    posts=[draw_posterior(data,logwt, rng = generator) for (data,logwt,logZ) in zip(datas,log_wts,log_evs)]
    if verbose:
        LOGGER.critical('Number of input samples: {0!s}'.format((str([len(x) for x in log_wts]))))
        LOGGER.critical('Expected number of samples from each input file {0!s}'.format((str([int(f*len(p)) for f,p in zip(fracs,posts)]))))
    bigpos=[]
    for post,frac in zip(posts,fracs):
      mask = generator(size=len(post))<frac
      bigpos.append(post[mask])
    result = np.concatenate([bigpos[i] for i in range(len(bigpos))], axis=None)
    if verbose:
        LOGGER.critical('Samples produced: {0:d}'.format(result.shape[0]))
    return result, log_total_evidence-np.log(len(datas))

def draw_N_posterior(data,log_wts, N, verbose=False):
    """
    Draw N samples from the input data, weighted by log_wt.
    For large N there may be repeated samples
    """
    if(N==0): return []
    log_cumsums=np.zeros(log_wts.shape[0]+1)
    log_cumsums[0]=-float('inf')
    for i,log_wt in enumerate(log_wts):
        log_cumsums[i+1]=logaddexp(log_cumsums[i], log_wt)

    us=np.log(uniform(size=N))
    idxs=np.digitize(us, log_cumsums)

    return data[idxs-1]

def draw_N_posterior_many(datas, Nlives, Npost, verbose=False):
    """
    Draw Npost samples from the posteriors represented by the
    (Nruns, Nsamples, Nparams)-shaped array datas, each sampled with
    the corresponding number of live points Nlive. The returned number
    of samples may not be exactly Npost due to rounding
    """
    # get log_evidences, log_weights.
    log_evs,log_wts=zip(*[compute_weights(data['logL'],Nlive) for data,Nlive in zip(datas, Nlives)])

    log_total_evidence=reduce(logaddexp, log_evs)
    Ns=[int(round(Npost*np.exp(log_ev-log_total_evidence))) for log_ev in log_evs]
    posts=[draw_N_posterior(data,logwt,N) for (data,logwt,N) in zip(datas,log_wts,Ns)]
    return vstack(posts).flatten()

def resample_mcmc_chain(chain, verbose=False, burnin=False):
    """
    Draw samples from the mcmc chains posteriors by redrawing
    each one of them against the Metropolis-Hastings rule
    """
    LOGGER.critical('Number of input samples: {0!s}'.format(chain.shape[0]))
    if burnin: chain = chain[chain.shape[0]/2-1:]
    # thin according to the autocorrelation length
    ACL = []
    for n in chain.dtype.names:
        if n != 'logL' and n != 'logPrior':
            ACL.append(acl(chain[n]))
    ACL = int(np.round(np.max(ACL)))

    LOGGER.critical('Measured autocorrelation length {0!s}'.format(str(ACL)))
    # thin and shuffle the chain
    chain = chain[::ACL]
    np.random.shuffle(chain)
    # compute the normalised log posterior density
    logpost = chain['logL']+chain['logPrior']
    # resample using a Metropolis-Hastings rule
    output =  [chain[0]]
    for i in range(1,chain.shape[0]):
        if logpost[i] - logpost[i-1] > np.log(uniform()):
            output.append(chain[i])

    output = np.array(output)
    LOGGER.critical('Returned number of samples {0!s}'.format(str(output.shape[0])))

    return output

def next_pow_two(n):
    """
    find the next power of 2 given n
    """
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorrelation(x, norm=True):
    """
    compute the autocorrelation function of an array
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def auto_window(taus, c):
    """
    Automated windowing procedure following Sokal (1989)
    """
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def acl(x, c=5.0):
    """
    Compute autocorrelation length for x
    """
    N = x.shape[0]
    acf = autocorrelation(x)
    T = 2.0*np.cumsum(acf)-1.0
    return T[auto_window(T, c)]
