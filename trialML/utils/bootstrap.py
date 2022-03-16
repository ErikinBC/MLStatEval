"""
Help functions for carrying out bootstrap inference
"""

import numpy as np
import bottleneck as bn
from scipy.stats import norm

from trialML.utils.utils import try_flatten
from trialML.utils.vectorized import quant_by_bool

# loo=threshold_loo;bs=threshold_bs;baseline=threshold;upper=True;axis=0
def bca_calc(loo, bs, baseline, alpha, axis=0, upper=True):
    """
    Calculate the bias and acceleration parameter from a LOO statistic and return the alpha-adjusted quantile

    Parameters
    -------
    loo:            Some array of LOO statistic values (# samples by # columns)
    bs:             The bootstraped value of the statistic  (# of columns by # of bootstraps)
    baseline:       Baseline statistic (# of columns)
    axis:           If axis == 1, then # of samples is transposed
    upper:          Whether we should take the 1-alpha adjusted quantile (True) or the alpha quantile (False)
    """
    assert axis in [0, 1], 'axis must be either 0 or 1'
    _, k = loo.shape
    k_bs, n_bs = bs.shape
    assert k == k_bs, 'error! loo.shape[1] == bs.shape[0]'
    k_bl = baseline.shape[0]
    assert k_bl == k, 'error! loo.shape[1] == basline.shape[0]'
    z_alpha = norm.ppf(alpha)
    loo_mu = bn.nanmean(loo, axis=axis)
    # allow for broadcasting
    baseline = np.expand_dims(baseline, axis=1-axis)  
    loo_mu = np.expand_dims(loo_mu, axis)
    # BCa calculation
    n_ineq = np.sum(bs < baseline, axis=1-axis)
    zhat0 = norm.ppf((n_ineq + 1) / (n_bs + 1))
    num = bn.nansum((loo_mu - loo)**3, axis=axis)
    den = 6* (bn.nansum((loo_mu - loo)**2, axis=axis))**(3/2)
    ahat = num / den
    alpha_adj = norm.cdf(zhat0 + (zhat0+z_alpha)/(1-ahat*(zhat0+z_alpha)))
    alpha_adj = try_flatten(alpha_adj)
    if upper:
        alpha_adj = 1 - alpha_adj
    bs_bool = ~np.isnan(bs)
    if axis == 0:
        bs, bs_bool = bs.T, bs_bool.T
    bca = quant_by_bool(data=bs, boolean=bs_bool, q=alpha_adj)
    return bca