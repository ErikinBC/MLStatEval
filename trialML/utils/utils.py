import os
import numpy as np
import pandas as pd

# Make a folder if it does not exist
def makeifnot(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Flatten an array if only a single dimension is not one
def try_flatten(x):
    xs = np.array(x.shape)
    ns1 = np.sum(xs > 1)
    if ns1 == 1:
        x = x.flatten()
    return x

# Row vector
def rvec(x):
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) == 1:
        return np.atleast_2d(x)
    else:
        return x

# Return as a column vector if 2d or less
def cvec(x):
    if len(x.shape) <= 2:
        z = rvec(x)
        if z.shape[0] == 1:
            z = z.T
        return z
    else:
        return x

# Check that float is between 0-1
def check01(x, inclusive=False):
    if inclusive:
        check = (x >= 0) & (x <= 1)
    else:
        check = (x > 0) & (x < 1)
    return check

# Check that array is all zeros or ones
def check_binary(x):
    ux = np.unique(x)
    check = all([z in [0,1] for z in ux])
    return check

# Convert input to a numpy array
def to_array(x):
    if not isinstance(x, np.ndarray):
        if isinstance(x, float) or isinstance(x, int):
            x = np.array([x])
        else:
            x = np.array(x)
    return x

# return a (1,1) or (1,) array/dataframe as a float
def array_to_float(x):
    if isinstance(x, list):
        x = np.array(x)
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            return x
        if max(x.shape) == 1:
            x = to_array(x).flatten()[0]
    return x

# Check if df is a DataFrame and extract columns and indices
def get_cn_idx(df):
    cn, idx = None, None
    if isinstance(df, pd.DataFrame):
        cn = list(df.columns)
        idx = df.index
    return cn, idx

# Provide a list of arguments and return the cn_idx for one of them
def df_cn_idx_args(*args):
    holder_cn, holder_idx = [], []
    for arg in args:
        cn, idx = get_cn_idx(arg)
        holder_cn.append(cn)
        holder_idx.append(idx)
    holder_cn = [cn for cn in holder_cn if cn is not None]
    holder_idx = [idx for idx in holder_idx if idx is not None]
    n_cn, n_idx = len(holder_cn), len(holder_idx)
    assert (n_cn <= 1) & (n_idx <= 1), 'only up to one argument should be a DataFrame!'
    if (n_cn == 1) | (n_idx == 1):
        cn, idx = holder_cn[0], holder_idx[0]
    else:
        cn, idx = None, None
    return cn, idx


# Convert operating threshold into column vector
def clean_threshold(threshold):
    if isinstance(threshold, float) or isinstance(threshold, int):
        threshold = np.array([threshold])
    if not isinstance(threshold, np.ndarray):
        threshold = np.array(threshold)
    threshold = cvec(threshold)
    return threshold

# Ensure labels and scores are column-vector arrays
def clean_y_s(y, s):
    yv = cvec(y)
    sv = cvec(s)
    return yv, sv

# Clean up labels, scores, and thresholds
def clean_y_s_threshold(y, s, threshold):
    # Put into column vectors
    y, s = cvec(y), cvec(s)
    # If thresold is a float, and s is a matrix, create an identical threshold for each column
    nc = s.shape[1]
    if isinstance(threshold, float) or isinstance(threshold, int):
        if (nc > 1):
            threshold = np.repeat(threshold, nc)
    cn, idx = get_cn_idx(threshold)  # If threshold is a DataFrame, extract information
    threshold = clean_threshold(threshold)  # Returns as column vector
    # Ensure operations broadcast
    y_shape = y.shape + (1,)
    t_shape = (1,) + threshold.shape
    y = y.reshape(y_shape)
    s = s.reshape(y_shape)
    threshold = threshold.reshape(t_shape)
    assert y.shape == s.shape, 'y and s must have the same shape!'
    assert len(s.shape) == len(threshold.shape), 'Number of dimensions must match between s/y and threshold'
    assert s.shape[1] == threshold.shape[1], 'Number of columns most align between threshold and shape'
    return cn, idx, y, s, threshold


# Save plotnine objects, delete existing file
def gg_save(fn, fold, gg, width, height):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.save(path, width=width, height=height, limitsize=False)

# Save a patchworklib grid
def grid_save(fn, fold, gg):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.savefig(path)

# Convert float to integer columns if possible
def df_float2int(df):
    assert isinstance(df, pd.DataFrame), 'df needs to be a DataFrame'
    cn_float = df.columns[(df.dtypes == float).values]
    if len(cn_float) > 0:
        idx_int = (df[cn_float] == df[cn_float].fillna(np.pi).astype(int)).all(0)
        cn_int = list(idx_int[idx_int == True].index)
        if len(cn_int) > 0:
            df[cn_int] = df[cn_int].astype(int)
    

# Interpolate values for some dataframe
def interp_df(df, cn_y, cn_x, target_y, *groups):
    # df=thresh_match; cn_x='val'; cn_y='thresh'; target_y=1.06; groups=('sim',)
    assert df.columns.isin([cn_y, cn_x]).sum() == 2, 'cn_y and cn_x must be columns in df'
    if len(groups) == 0:
        df = df.assign(gg='group1')
        groups = ['group1']
    else:
        groups = list(groups)
    # Sort data and reset index
    df = df.sort_values(groups+[cn_y]).reset_index(drop=True)
    # Get value just above and below target_y
    df = df.assign(is_below=lambda x: np.where(x[cn_y] < target_y, 'below', 'above'))
    t1 = df.groupby(groups+['is_below']).apply(lambda x: x.loc[x[cn_y].idxmax()])
    t2 = df.groupby(groups+['is_below']).apply(lambda x: x.loc[x[cn_y].idxmin()])
    t1 = t1.reset_index(drop=True).query('is_below=="below"')
    t2 = t2.reset_index(drop=True).query('is_below=="above"')
    # Pivot wide
    dat = pd.concat(objs=[t1, t2], axis=0)
    dat = dat.pivot_table([cn_y, cn_x], groups, 'is_below')
    dat.columns = ['_'.join(col).strip() for col in dat.columns.values]
    cn_x_below = cn_x+'_below'
    cn_x_above = cn_x+'_above'
    cn_y_below = cn_y+'_below'
    cn_y_above = cn_y+'_above'
    cn_x_interp = cn_x+'_interp'
    # Calculate the slope
    dat = dat.assign(slope=lambda x: (x[cn_y_above]-x[cn_y_below])/(x[cn_x_above]-x[cn_x_below]))
    dat = dat.assign(interp=lambda x: (x[cn_y_above]-target_y)/x['slope']+x[cn_x_above] )
    dat = dat.reset_index().rename(columns={'interp':cn_x_interp})
    dat = dat[groups+[cn_x_interp]]
    return dat

# # No different between two lists
# def no_diff(x, y):
#     check1 = len(np.setdiff1d(x, y)) == 0
#     check2 = len(np.setdiff1d(y, x)) == 0
#     return check1 & check2

# # Round up to some factor
# def round_up(num, factor):
#     w, r = divmod(num, factor)
#     return factor*(w + int(r>0))


# # Apply a column-wise quantile 
# def quantile_mapply(arr, q, axis=0):  
#     # arr=power_bs.copy();q=alpha_adj.copy()
#     j = int(np.where(axis == 0, 1, 0))
#     assert arr.shape[j] == q.shape[j]
#     if axis == 0:
#         arr = arr.copy().T
#         q = q.copy().T                
#     holder = np.zeros(q.shape)
#     for k in range(arr.shape[0]):
#         holder[k] = np.quantile(arr[k], q[k])
#     if axis == 0:
#         holder = holder.T
#     return holder

