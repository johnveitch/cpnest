import numpy as np
from numpy import logaddexp, vstack
from numpy.random import uniform
from functools import reduce

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

def draw_posterior(data, log_wts, verbose=False):
    """Draw points from the given data (of shape (Nsamples, Ndim))
    with associated log(weight) (of shape (Nsamples,)). Draws uniquely so
    there are no repeated samples"""
    maxWt=max(log_wts)
    normalised_wts=log_wts-maxWt
    selection=[n > np.log(uniform()) for n in normalised_wts]
    idx=list(filter(lambda i: selection[i], range(len(selection))))
    return data[idx]
    
def draw_posterior_many(datas, Nlives, verbose=False):
    """Draw samples from the posteriors represented by the
    (Nruns, Nsamples, Nparams)-shaped array datas, each sampled with
    the corresponding Nlive number of live points. Will draw without repetition,
    and weight according to the evidence in each input run"""
    # list of log_evidences, log_weights
    log_evs,log_wts=zip(*[compute_weights(data['logL'],Nlive) for data,Nlive in zip(datas, Nlives)])
    if verbose: print('Computed log_evidences: {0!s}'.format((str(log_evs))))

    log_total_evidence=reduce(logaddexp, log_evs)
    log_max_evidence=max(log_evs)
    #print 'evidences: %s'%(str(log_evs))
    fracs=[np.exp(log_ev-log_max_evidence) for log_ev in log_evs]
    if verbose: print('Relative weights of input files: {0!s}'.format((str(fracs))))
    Ns=[fracs[i]/len(datas[i]) for i in range(len(fracs))]
    Ntot=max(Ns)
    fracs=[n/Ntot for n in Ns]
    if verbose: print('Relative weights of input files taking into account their length: {0!s}'.format((str(fracs))))

    posts=[draw_posterior(data,logwt) for (data,logwt,logZ) in zip(datas,log_wts,log_evs)]
    if verbose: print('Number of input samples: {0!s}'.format((str([len(x) for x in log_wts]))))
    if verbose: print('Expected number of samples from each input file {0!s}'.format((str([int(f*len(p)) for f,p in zip(fracs,posts)]))))
    bigpos=[]
    for post,frac in zip(posts,fracs):
      mask = uniform(size=len(post))<frac
      bigpos.append(post[mask])
    result = vstack(bigpos).flatten()
    if verbose: print('Samples produced: {0:d}'.format(result.shape[0]))
    return result
    
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

def draw_posterior(data, log_wts, verbose=False):
    """Draw points from the given data (of shape (Nsamples, Ndim))
        with associated log(weight) (of shape (Nsamples,)). Draws uniquely so
        there are no repeated samples"""
    maxWt=max(log_wts)
    normalised_wts=log_wts-maxWt
    selection=[n > np.log(uniform()) for n in normalised_wts]
    idx=list(filter(lambda i: selection[i], range(len(selection))))
    return data[idx]

def redraw_mcmc_chain(chain, verbose=False, burnin=True):
    """
    Draw samples from the mcmc chains posteriors by redrawing
    each one of them against the Metropolis-Hastings rule
    """
    if verbose: print('Number of input samples: {0!s}'.format(chain.shape[0]))
    if burnin: chain = chain[chain.shape[0]/2-1:]
    ACL = []
    for n in chain.dtype.names:
        if n != 'logL' and n != 'logPrior':
            acf = 2*np.cumsum(autocorrelation(chain[n]))-1
            ACL.append(acl(acf))
    ACL = int(np.round(np.max(ACL)))

    if verbose: print('Measured autocorrelation length {0!s}'.format(str(ACL)))
    chain = chain[::ACL]
    logpost = chain['logL']+chain['logPrior']
    # reweight using the MH rule
    us = np.log(uniform(size=len(logpost)-1))
    dlp = np.diff(logpost)
    (idx,) = np.where(dlp > us)
    chain = chain[idx+1]
    if verbose: print('Returned number of samples {0!s}'.format(str(chain.shape[0])))

    return chain

def autocorrelation (x) :
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x-np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size/2]/np.sum(xp**2)

def acl(acf):
    for i in range(len(acf)/2):
        if acf[i] < i/5.0:
            return acf[i]+1
