# Functions that do vectorized operations for different summary statistics
import numpy as np
from funs_support import cvec

"""
Calculates the LOO quantile (returns same size as data)
"""
def loo_quant_by_bool(data, boolean, q, interpolate='linear'):
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
    ns, nr, nc = x.shape
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
    ns, nr, nc = x.shape
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
