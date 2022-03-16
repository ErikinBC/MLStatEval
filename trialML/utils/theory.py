"""
Script which contains helper functions for Gaussian mixture theory
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar
from trialML.utils.utils import to_array, array_to_float, df_cn_idx_args

def power_binom(spread, n_trial, gamma, alpha, gamma0=None):
    """
    spread:             Null hypothesis spread (gamma - gamma_{H0})
    n_trial:            Expected number of trial points (note this is class specific!)
    """
    cn, idx = df_cn_idx_args(spread, n_trial)
    # Allow for vectorization
    spread, n_trial = to_array(spread), to_array(n_trial)
    # Return to scalar if necessary
    spread, n_trial = array_to_float(spread), array_to_float(n_trial)
    assert np.all(spread > 0) & np.all(spread < gamma), 'spread must be between (0, gamma)'
    if gamma0 is None:
        gamma0 = gamma - spread
    else:
        spread = gamma - gamma0
    sig0 = np.sqrt( gamma0*(1-gamma0) / n_trial )
    sig = np.sqrt( gamma*(1-gamma) / n_trial )
    z_alpha = norm.ppf(1-alpha)
    power = norm.cdf( (spread - sig0*z_alpha) / sig )
    power = array_to_float(power)
    if isinstance(cn, list):
        power = pd.DataFrame(power, columns = cn, index=idx)
    return power


def oracle_auroc(mu1, mu0, sd1, sd0):
    auroc = norm.cdf((mu1 - mu0) / np.sqrt(sd1**2 + sd0**2))
    return auroc

# Map a threshold to the performance measure
def threshold_to_sensitivity(threshold, mu1, sd1):
    # Convert inputs to arrays for vectorization
    threshold = to_array(threshold)
    mu1, sd1 = to_array(mu1), to_array(sd1)
    sens = norm.cdf( (mu1 - threshold) / sd1 )
    sens = array_to_float(sens)
    return sens

def threshold_to_specificity(threshold, mu0, sd0):
    threshold = to_array(threshold)
    mu0, sd0 = to_array(mu0), to_array(sd0)
    spec = norm.cdf( (threshold - mu0) / sd0 )
    spec = array_to_float(spec)
    return spec

def threshold_to_precision(threshold, mu1, mu0, sd1, sd0, p):
    sens = threshold_to_sensitivity(threshold, mu1, sd1)
    fpr = 1 - threshold_to_specificity(threshold, mu0, sd0)
    prec = p*sens / (p*sens + (1-p)*fpr)
    return prec

# For a given target of a performance measure, find the corresponding threhsold
def sensitivity_to_threshold(target, mu1, sd1):
    target = to_array(target)
    mu1, sd1 = to_array(mu1), to_array(sd1)
    threshold = mu1 + sd1*norm.ppf(1-target)
    return threshold

def specificity_to_threshold(target, mu0, sd0):
    target = to_array(target)
    mu0, sd0 = to_array(mu0), to_array(sd0)
    threshold = mu0 + sd0*norm.ppf(target)
    return threshold

# Wrapper to be called by root finder
def err_precision_target(threshold, target, mu1, mu0, sd1, sd0, p):
    prec = threshold_to_precision(threshold, mu1, mu0, sd1, sd0, p)
    err = target - prec
    return err

def imills(threshold, mu, sd):
    z = (mu - threshold) / sd
    ratio = norm.pdf(z) / norm.cdf(z)
    return ratio


def err_imills(threshold, mu1, mu0, sd1, sd0):
    lam1 = imills(threshold, mu1, sd1)
    lam0 = imills(threshold, mu0, sd0)
    err = lam1 - (sd1 / sd0)*lam0
    return err


def root_finding_range(mu1, mu0, sd1, sd0, n=4):
    """Wrapper to help find a reasonable search range over two Gaussians
    n:          # of sds away from means to search
    """
    sd_max = max(sd1, sd0)
    lb = mu0 - n*sd_max
    ub = mu1 + n*sd_max
    return lb, ub

"""Wrapper to call in root_scalar and check for convergence
f:          Function to find root over first parameter
x0:         Lower-bound start point
x1:         Upper-bound start point
lb:         Lower-bound search
ub:         Upper-bound search
method:     Root finding method
xtol:       Root finding tolerance
"""
def root_wrapper(f, args, x0, x1, lb, ub, method='secant', xtol=1e-5):
    assert isinstance(args, tuple), 'args must be a tuple'
    assert lb < ub, 'lower bound must be less than upper-bound'
    optim = root_scalar(f=f, args=args, x0=x0, x1=x1, method=method, bracket=(lb,ub), xtol=xtol)
    assert optim.converged, 'Root finder for did not converge!'
    val = optim.root
    return val


def precision_threshold_range(mu1, mu0, sd1, sd0, p):
    """Finds the theoretical precision limits"""
    var_eq = (sd1 == sd0)
    if var_eq:
        # Monotonically increasing from p to 1
        # (-infty, t*):(p, p*), (t*, infty):(p*, 1)
        prec_min, prec_max = p, 1
        thresh_local = None
    else:
        # Find the point where monotonicity changes
        lb, ub = root_finding_range(mu1, mu0, sd1, sd0)
        thresh_local = root_wrapper(f=err_imills, args=(mu1, mu0, sd1, sd0), x0=mu0, x1=mu1, lb=lb, ub=ub, method='secant')
        prec_local = threshold_to_precision(thresh_local, mu1, mu0, sd1, sd0, p)
    if not var_eq:
        if sd0 > sd1: 
            # Monotonically increasing from p to p_max, then monotonically decreasing from p_max to 0
            # (-infty, t*):(p, prec_max), (t*, infty):(prec_max, 0)
            assert prec_local > p, 'huh? prec_max should be greater than p!'
            prec_min = 0
            prec_max = prec_local
        else:   # sd1 > sd0
            # Monotonically decreasing from p to p_min, then monotonically increasing from p_min to 1
            # (-infty, t_low):(p, p_low), (t_low,t*):(p_low, p*), (t*,infty):(p*,1)
            assert prec_local < p, 'huh? prec_min should be less than p!'
            prec_min = prec_local
            prec_max = 1
    return prec_min, prec_max, thresh_local


def precision_to_threshold(target, mu1, mu0, sd1, sd0, p, w=0.25, xtol=1e-5):
    """
    Use root finding to map a target precision to an oracle threshold
    Depending on the relative variances, not all solutions exist
    Call precision_threshold_range() to get possible range first

    target:             precision target (may result in assertion error if outside feasable range)
    mu{j}:              Mean of class {j}
    sd{j}:              Standard deviation of class {j}
    p:                  Class balance P(y == 1)
    w:                  How to weight x0/x1 relative to (lb, ub)
    """
    assert (w > 0) & (w < 0.5), 'If weight is specificied need to be between (0,0.5)'
    # Get the precision ranges
    prec_min, prec_max, thresh_change = precision_threshold_range(mu1, mu0, sd1, sd0, p)
    assert (target > prec_min) & (target < prec_max), 'error! precision target must be between (%0.3f, %0.3f)' % (prec_min, prec_max)
    # Threshold search will depend on variance conditions
    var_eq = (sd1 == sd0)
    lb, ub = root_finding_range(mu1, mu0, sd1, sd0)
    if var_eq:
        # Two means are reasonable start point
        x0, x1 = mu0, mu1
    else:
        if sd0 > sd1:
            if (target >= p) & (target <= prec_max):
                # If target is between [p, prec_max], pick a smaller thresold
                ub = thresh_change
            else:
                # If the target is less than p, pick a larger threshold
                lb = thresh_change
        else:  # sd1 > sd0
            if target > p:
                # If target is between (p, 1], pick a larget threshold
                lb = thresh_change
            else:
                # If starget is between [prec_min, p], pick a smaller threshold
                ub = thresh_change
        # Adjust trial points to something between lb/ub
        x0 = (1-w)*lb + w*ub
        x1 = w*lb + (1-w)*ub
    thresh_hat = root_wrapper(f=err_precision_target, args=(target, mu1, mu0, sd1, sd0, p), x0=x0, x1=x1, lb=lb, ub=ub, method='brentq')
    prec_hat = threshold_to_precision(thresh_hat, mu1, mu0, sd1, sd0, p)
    assert np.abs(prec_hat - target) < xtol, 'Tolereance not found!'
    return thresh_hat
    