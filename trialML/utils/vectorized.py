# Functions that do vectorized operations for different summary statistics
import numpy as np
import pandas as pd
from trialML.utils.utils import cvec

def vec_arange(starts, lengths):
    """Create multiple aranges with [start1, start2,...] and [length1, length2,...]
    """
    # Repeat start position index length times and concatenate
    cat_start = np.repeat(starts, lengths)
    # Create group counter that resets for each start/length
    cat_counter = np.arange(np.sum(lengths)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    # Add group counter to group specific starts
    cat_range = cat_start + cat_counter
    return cat_range


def find_empirical_precision(y, s, target, ret_idx=False):
    """Find infiimum threshold that gets a precision target

    Assumes that y,s have dimensions that correspond to: (# of observations) x (# of columns) x (# of simulations). This is important for take_along_axis and ridx
    """
    y, s = cvec(y), cvec(s)
    assert y.shape == s.shape, 'y and s need to be the same shape'
    n, _ = s.shape[:2]
    # number of observations needs to march s.shape
    ridx = np.arange(1,n+1)
    nd = len(y.shape) - len(ridx.shape)
    if nd > 0:
        ridx = np.expand_dims(ridx, list(range(nd))).T
    idx_s = np.argsort(-s, axis=0)
    y_sort = np.take_along_axis(y, idx_s, 0)
    s_sort = np.take_along_axis(s, idx_s, 0)
    tps = np.cumsum(y_sort, axis=0)
    ppv = tps / ridx
    # We want to apply argmin on only valid thresholds
    hits_target = ppv >= target
    s_adjust = np.where(hits_target, s_sort, np.inf)
    idx_thresh = np.argmin(s_adjust, axis=0)
    if ret_idx:
        return idx_thresh
    any_thresh = np.any(hits_target, axis=0)
    thresh_s = np.take_along_axis(s_sort, idx_thresh[None], 0)
    # If no target is met, no threshold
    any_thresh = any_thresh.reshape(thresh_s.shape)
    thresh_s = np.where(any_thresh, thresh_s, np.nan)
    thresh_s = np.squeeze(thresh_s)
    return thresh_s


def gt2inf(x, t, r):
    """
    Compare array x >= t, where condition is met, put r, otherwise inf
    """
    z = np.where(x >= t, r, np.inf)
    return z

def loo_precision(y, s, target):
    """calculates the leave-one-out precision using an interpolation method. Since the empirical infimum has to occur on y==1, and the score below it has to be below the target....

    This can be done rapidly since precision shifts only for the point around threshold
    """
    y, s = cvec(y), cvec(s)
    assert y.shape == s.shape, 'y and s need to be the same shape'
    n, k = s.shape[:2]
    # number of observations needs to march s.shape
    ridx = np.arange(1,n+1)
    nd = len(y.shape) - len(ridx.shape)
    if nd > 0:
        ridx = np.expand_dims(ridx, list(range(nd))).T
    idx_s = np.argsort(-s, axis=0)
    y_sort = np.take_along_axis(y, idx_s, 0)
    s_sort = np.take_along_axis(s, idx_s, 0)
    tps = np.cumsum(y_sort, axis=0)
    fps = np.cumsum(1-y_sort, axis=0)
    ppv = tps / (tps + fps)
    # Three conditions:
    # i) 1 is dropped above threshold
    tps_1up = np.maximum(tps - 1, 0)
    ppv_1up = tps_1up / (tps_1up + fps)
    idx_1up = np.argmin(gt2inf(ppv_1up, target, s_sort), axis=0)[None]
    thresh_1up = np.take_along_axis(s_sort, idx_1up, axis=0)
    check_1up = np.take_along_axis(ppv_1up, idx_1up, axis=0) >= target
    # (ii) 0 is dropped above threshold
    fps_0up = np.maximum(fps - 1, 0)
    ppv_0up = tps / (tps + fps_0up)
    idx_0up = np.argmin(gt2inf(ppv_0up, target, s_sort), axis=0)[None]
    thresh_0up = np.take_along_axis(s_sort, idx_0up, axis=0)
    check_0up = np.take_along_axis(ppv_0up, idx_0up, axis=0) >= target
    # (iii) 1/0 is dropped below threshold results in original value
    idx_01b = find_empirical_precision(y, s, target, ret_idx=True)[None]
    thresh_01b = np.take_along_axis(s_sort, idx_01b, axis=0)
    check_01b = np.take_along_axis(ppv, idx_01b, axis=0) >= target
    # Calculate number of times condition is met
    n_1up = np.take_along_axis(tps, idx_01b, axis=0)
    n_0up = (idx_01b+1) - n_1up
    n_01b = n - (n_1up + n_0up)
    # Repeat array to match this
    n_rep = np.r_[n_1up, n_0up, n_01b].T
    assert np.sum(n_rep, axis=1).var() == 0, 'n_rep should be the same for each'
    s_rep = np.r_[thresh_1up, thresh_0up, thresh_01b].T
    # If target is not met, return nan
    check_rep = np.r_[check_1up, check_0up, check_01b].T
    s_rep[~check_rep] = np.nan
    s_loo = np.repeat(s_rep.flat, n_rep.flat).reshape([k, n]).T
    return s_loo


def loo_quant_by_bool(data, boolean, q):
    """Calculates the LOO quantile (returns same size as data)
    """
    # (i) Prepare data
    x = np.where(boolean, data, np.nan)  # For False values, assign nan
    ndim = len(x.shape)
    assert ndim <= 3, 'Function only works for up to three dimensions'
    if ndim == 1:
        boolean = cvec(boolean)
        x = cvec(x)
    if ndim == 2:
        shape = (1,) + x.shape
        boolean = boolean.reshape(shape)
        x = x.reshape(shape)
    assert x.shape == boolean.shape
    ns, nr, nc = x.shape
    sidx = np.repeat(range(ns), nc)
    cidx = np.tile(range(nc), ns)

    # (ii) Sort by columns
    x = np.sort(x, axis=1)
    n = np.sum(boolean, axis=1)  # Number of non-missing rows    
    n_flat = n.flatten()

    # (iii) Do LOO calculations
    n2 = n - 2
    ridx = n2*q
    lidx = np.clip(np.floor(ridx).astype(int),0,n.max())
    uidx = np.clip(np.ceil(ridx).astype(int),0,n.max())
    frac = ridx - lidx
    l = lidx.flatten()  # For [i,:,j] indexing
    u = uidx.flatten()
    reshape = lidx.shape  # To return to shape after flatten
    ndim = ns * nc
    starts = np.repeat(0, ndim)
    # Holder
    q_loo = np.where(np.isnan(x), np.nan, 0)

    # (i) Values up to (and including) the lower bound
    xl = x[sidx, l+1, cidx].reshape(reshape)
    xu = x[sidx, u+1, cidx].reshape(reshape)
    xd = xu - xl
    x_interp = xl + xd*frac
    loo_1 = np.repeat(x_interp.flat, l+1)
    # Prepare indices
    idx_s = np.repeat(sidx, l+1)
    idx_c = np.repeat(cidx, l+1)
    idx_r = vec_arange(starts, l+1)
    assert idx_s.shape == idx_c.shape == idx_r.shape
    q_loo[idx_s, idx_r, idx_c] = loo_1
    
    # (ii) upped-bound removed
    xl = x[sidx, l, cidx].reshape(reshape)
    xu = x[sidx, u+1, cidx].reshape(reshape)
    xd = xu - xl
    x_interp = xl + xd*frac
    loo_2 = np.repeat(x_interp, 1)
    q_loo[sidx,l+1,cidx] = loo_2

    # (iv) Values above the upper bound
    xl = x[sidx, l, cidx].reshape(reshape)
    xu = x[sidx, u, cidx].reshape(reshape)
    xd = xu - xl
    x_interp = xl + xd*frac
    n_pos_u = u + 1 + (l==u)
    n_left = n_flat - n_pos_u
    loo_3 = np.repeat(x_interp.flat, n_left)    
    idx_s = np.repeat(sidx, n_left)
    idx_c = np.repeat(cidx, n_left)
    idx_r = vec_arange(n_pos_u, n_flat-n_pos_u)
    q_loo[idx_s,idx_r,idx_c] = loo_3
    
    # Return imputed value
    return q_loo



"""Find the quantile (linear interpolation) for specific rows of each column of data
data:           np.array of data (nsim x nobs x ncol)
boolean:        np.array of which (i,j) positions to include in calculation
q:              Quantile target
"""
# data=df_s_val[3];boolean=(df_y_val[3]==1);q=0.5;interpolate='linear'
def quant_by_bool(data, boolean, q, interpolate='linear'):
    assert interpolate in ['linear', 'lower', 'upper']
    # (i) Prepare data
    x = np.where(boolean, data, np.nan)  # For False values, assign nan
    ndim = len(x.shape)
    assert ndim <= 3, 'Function only works for up to three dimensions'
    if ndim == 1:
        boolean = cvec(boolean)
        x = cvec(x)
    if ndim == 2:
        shape = (1,) + x.shape
        boolean = boolean.reshape(shape)
        x = x.reshape(shape)
    assert x.shape == boolean.shape
    ns, _, nc = x.shape
    sidx = np.repeat(range(ns), nc)
    cidx = np.tile(range(nc), ns)

    # (ii) Sort by columns
    x = np.sort(x, axis=1)
    n = np.sum(boolean, axis=1)  # Number of non-missing rows

    # (iii) Find the row position that corresponds to the quantile
    ridx = q*(n-1)
    lidx = np.clip(np.floor(ridx).astype(int),0,n.max())
    uidx = np.clip(np.ceil(ridx).astype(int),0,n.max())    
    frac = ridx - lidx

    # (iv) Return depends on method
    reshape = lidx.shape
    if ns == 1:
        reshape = (nc, )  # Flatten if ns == 1
    q_lb = x[sidx, lidx.flatten(), cidx].reshape(reshape)
    if interpolate == 'lower':
        return q_lb
    q_ub = x[sidx, uidx.flatten(), cidx].reshape(reshape)
    if interpolate == 'upper':
        return q_ub
    # do linear interpolation
    dq = q_ub - q_lb
    q_interp = q_lb + dq*frac
    q_interp = q_interp.reshape(reshape)
    return q_interp

# data = np.random.randn(100, 3); q=[0.5, 0.5, 0.5]
def quant_by_col(data, q):
    q = np.array(q)
    assert len(data.shape) == 2
    assert data.shape[1] == len(q), 'q needs to align with data column dimension'
    
    # (ii) Sort by columns
    x = np.sort(data, axis=0)
    n, c = x.shape
    cidx = np.arange(c)

    # (iii) Find the row position that corresponds to the quantile
    ridx = q*(n-1)
    lidx = np.floor(ridx).astype(int)
    uidx = np.ceil(ridx).astype(int)
    frac = ridx - lidx

    # (iv) Linear interpolation
    q_lb = x[lidx, cidx]
    q_ub = x[uidx, cidx]
    dq = q_ub - q_lb
    q_interp = q_lb + dq*frac
    return q_interp
