import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, skewnorm
from scipy.optimize import minimize_scalar, root_scalar
from statsmodels.stats.proportion import proportion_confint as prop_CI

# Function to find oracle threshold for Gaussian dist
def prec_gauss(thresh, mu1, p, mu0=0, s1=1, s0=1):
    assert mu1 > mu0
    z1 = (mu1 - thresh)/s1
    z0 = (mu0 - thresh)/s0
    term1 = norm.cdf(z1)*p
    term2 = norm.cdf(z0)*(1-p)
    ppv = term1/(term1+term2)
    return ppv

def err_prec(thresh, target, mu, p, square=False):
    ppv = prec_gauss(thresh, mu, p)
    err = ppv - target
    if square:
        err = err**2
    return err

# Function to plot threshold/precision trade-off
def pr_curve(mu1, p, mu0=0, s1=1, s0=1, n_points=100, alpha=0.001):
    assert mu1 > mu0, 'mu1 must be > mu0'
    z_alpha = norm.ppf(1-alpha)
    # (i) Plotting ranges
    lb1, lb2 = mu0 - z_alpha*s0, mu1 - z_alpha*s1
    lb = min(lb1, lb2)
    ub1, ub2 = mu0 + z_alpha*s0, mu1 + z_alpha*s1
    ub = max(ub1, ub2)
    thresh_seq = np.linspace(lb, ub, n_points)
    ppv = prec_gauss(thresh_seq, mu1=mu1, mu0=mu0, s1=s1, s0=s0, p=p)
    recall = norm.cdf((mu1 - thresh_seq)/s1)
    res = pd.DataFrame({'thresh':thresh_seq, 'ppv':ppv, 'recall':recall})
    return res


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


