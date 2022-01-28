import numpy as np
import pandas as pd
from scipy.stats import rankdata, skewnorm
from scipy.optimize import minimize_scalar
from statsmodels.stats.proportion import proportion_confint as prop_CI

# Probability that one skewnorm is less than another
def sn_ineq(mu, skew, scale, alpha, n_points):
    dist_y1 = skewnorm(a=skew, loc=mu, scale=scale)
    dist_y0 = skewnorm(a=skew, loc=0, scale=scale)
    x_seq = np.linspace(dist_y0.ppf(alpha), dist_y1.ppf(1-alpha), n_points)
    dx = x_seq[1] - x_seq[0]
    prob = np.sum(dist_y0.cdf(x_seq)*dist_y1.pdf(x_seq)*dx)
    return prob

# Find the mean of a skewnorm that achieves a certain AUROC
def find_auroc(auc, skew, scale=1, alpha=0.001, n_points=100):
    optim = minimize_scalar(fun=lambda mu: (auc - sn_ineq(mu, skew, scale, alpha, n_points))**2,method='brent')
    assert optim.fun < 1e-10
    mu_star = optim.x
    return mu_star

# Fast method of calculating AUROC
def auc_rank(y, s):
    n1 = sum(y)
    n0 = len(y) - n1
    den = n0 * n1
    num = sum(rankdata(s)[y == 1]) - n1*(n1+1)/2
    auc = num / den
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
def get_CI(df, cn_n, den, method='beta', alpha=0.05):
    tmp = pd.concat(prop_CI(df[cn_n], den, method=method),axis=1)
    tmp.columns = ['lb', 'ub']
    df = pd.concat(objs=[df, tmp], axis=1)
    return df


"""Find the quantile (linear interpolation) for specific rows of each column of data
data:           np.array of data to quantiles of
boolean:        np.array of which (i,j) positions to include in calculation
q:              Quantile target
"""
def fast_quant_by_bool(data, boolean, q):
    # For False values, assign nan
    x = np.where(boolean, data, np.nan)
    # Sort
    x = np.sort(x, axis=0)
    n = np.sum(boolean, 0)
    # Find the row position that corresponds to the quantile
    ridx = q*(n-1)
    lidx = np.floor(ridx).astype(int)
    uidx = np.ceil(ridx).astype(int)
    frac = ridx - lidx
    cidx = np.arange(x.shape[1])
    # calculate lower/upper bound of quantile
    q_lb = x[lidx, cidx]
    q_ub = x[uidx, cidx]
    # do linear interpolation
    dq = q_ub - q_lb
    q_interp = q_lb + dq*frac
    return q_interp

# def quant_by_bool(data, boolean, q):
#     kk = data.shape[1]
#     holder = np.zeros(kk)
#     for k in range(kk):
#         holder[k] = np.quantile(data[:,k][boolean[:,k]], q)
#     return holder
# from timeit import timeit
# q, n, k, nsim = 0.75, 100, 1000, 100
# data = np.random.randn(n,k)
# boolean = np.random.binomial(1,0.5,(n,k)) == 1
# assert np.all(quant_by_bool(data, boolean, q) == fast_quant_by_bool(data, boolean, q))
# timeit(stmt='quant_by_bool(data, boolean, q)', number=nsim, globals=globals())
# timeit(stmt='fast_quant_by_bool(data, boolean, q)', number=nsim, globals=globals())