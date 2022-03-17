import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import rankdata, skewnorm
from scipy.optimize import root_scalar
from statsmodels.stats.proportion import proportion_confint as prop_CI

import warnings
warnings.filterwarnings("ignore", message="overflow encountered in _beta_isf")

# Internal modules
from trialML.utils.utils import cvec, check01, array_to_float

def umbrella_thresh(n, target, alpha, upper=True):
    """
    Function to implement Neyman-Pearson umbrella rank choice. WARNING, if n is too large, overflow error possible

    Parameters
    ----------
    n:          sample size (class specific)
    k:          performance target
    alpha:      type-I error
    upper:      should k be for for the ascending or descending rank order?
    
    Returns
    -------
    Row index for sorted values
    """
    assert check01(target), 'target needs to be between (0,1)'
    assert check01(alpha), 'alpha needs to be between (0,1)'
    is_float = (isinstance(n, int) or isinstance(n, float))
    is_array = (isinstance(n, np.ndarray) | isinstance(n, pd.Series))
    assert is_float or is_array, 'n must be either a float/int or an array'
    if is_float:
        n_seq = np.array([n])
    else:
        n_seq = pd.Series(np.unique(n))
    di_holder = {}
    for n1 in n_seq:
        rank_seq = np.arange(n1+1)
        rank_pdf = np.array([comb(n1, l, True)*((1-target)**l)*(target**(n1-l)) for l in rank_seq])
        rank_cdf = np.array([rank_pdf[l:].sum() for l in rank_seq])
        res = pd.DataFrame({'rank':rank_seq, 'pdf':rank_pdf, 'cdf':rank_cdf, 'delt':1-rank_cdf})
        rmax = res[res['delt'] <= alpha]['rank'].argmax()
        if upper:
            rmax = n1 - rmax
        else:  # Flip
            rmax -= 1  # For numpy indexing
        di_holder[n1] = rmax
    rvec = np.array(pd.Series(n).map(di_holder))
    rvec = array_to_float(rvec)
    return rvec


# Probability that one skewnorm is less than another
def sn_ineq(mu, skew, scale, alpha, n_points):
    dist_y1 = skewnorm(a=skew, loc=mu, scale=scale)
    dist_y0 = skewnorm(a=skew, loc=0, scale=scale)
    x_seq = np.linspace(dist_y0.ppf(alpha), dist_y1.ppf(1-alpha), n_points)
    dx = x_seq[1] - x_seq[0]
    prob = np.sum(dist_y0.cdf(x_seq)*dist_y1.pdf(x_seq)*dx)
    return prob

# Find the mean of a skewnorm that achieves a certain AUROC
def find_auroc(auc, skew, scale=1, alpha=0.001, n_points=100, bound=100):
    optim = root_scalar(f=lambda mu: (auc - sn_ineq(mu, skew, scale, alpha, n_points)), x0=0, x1=0.1, method='secant', bracket=(-bound,+bound))
    assert optim.flag == 'converged', 'optimization did not converge!'
    mu_star = optim.root
    return mu_star


# Fast method of calculating AUROC
def auc_rank(y, s):
    y, s = cvec(y), cvec(s)
    assert y.shape == s.shape, 'y and s need to have the same shape'
    n1 = np.sum(y, axis=0)
    n0 = len(y) - n1
    den = n0 * n1
    num = np.sum(rankdata(s, axis=0) * (y == 1), 0)
    num -= n1*(n1+1)/2
    auc = num / den
    if len(auc) == 1:
        auc = auc[0]  # Return as float is original are arrays
    return auc

# SKLearn wrapper to calculate empirical ROC
def emp_roc_curve(y, s):
    s1, s0 = s[y == 1], s[y == 0]
    s1 = np.sort(np.unique(s1))
    s0 = np.sort(np.unique(s0))
    thresh_seq = np.flip(np.sort(np.append(s1, s0)))
    # Get range of sensitivity and specificity
    sens = [np.mean(s1 >= thresh) for thresh in thresh_seq]
    spec = [np.mean(s0 < thresh) for thresh in thresh_seq]
    # Store
    res = pd.DataFrame({'thresh':thresh_seq, 'sens':sens, 'spec':spec})
    return res

# Add on CIs to a dataframe
def get_CI(df, cn_num, cn_den, method='beta', alpha=0.05):
    """
    Add on lb/ub confidence interval columns

    df:             Some pd.DataFrame
    cn_num:         Number of successes
    cn_den:         Number of chances
    method:         Binomial confidence method
    alpha:          Type-I error rate
    """
    assert isinstance(df, pd.DataFrame), 'df needs to be a DataFrame'
    tmp = pd.concat(prop_CI(df[cn_num], df[cn_den], alpha=alpha, method=method),axis=1)
    tmp.columns = ['lb', 'ub']
    df = pd.concat(objs=[df, tmp], axis=1)
    return df


