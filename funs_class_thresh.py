# Class needed to calibrate threshold for specific performance target
import numpy as np
import pandas as pd
from scipy.stats import norm
from funs_support import cvec

lst_m = ['sens','spec','prec']
lst_method = ['quantile', 'bs-q', 'bs-bca']

XX = np.arange(10).reshape([2,5]).T
YY = np.tile([0,1],5).reshape([2,5]).T
# Quantile specific projects...

# Class specific categories for each performance function
class sensitivity():
    def __init__(self, mu=1):
        mu = self.mu

    @staticmethod
    def stat(y, s, t):
        yhat = np.where(s >= t, 1, 0)
        tps = np.sum((y == yhat) * (y == 1), 0)
        ps = y.sum(0)
        sens = tps / ps
        return sens

    def learn_thresh(y, s, gamma, method):
        assert method in lst_method
        assert (gamma >= 0) & (gamma <= 1)
        if method == 'quantile':
            s[y == 1]
            



class tools_thresh():
    """
    m:          Performance measure
    mu:         Mean of positive class N(mu, 1)
    p:          Probability of an observation being a positive class
    alpha:      Type-I error of the null hypothesis test
    """
    def __init__(self, m, alpha=0.05, mu=1, p=0.5):
        # (i) Performance measure
        assert m in lst_m, 'm needs be one of: %s' % lst_m
        
        # (ii) Statistical testing
        assert (alpha > 0) and (alpha < 1), 'Type-I error must be between 0 to 1'
        self.alpha = alpha
        self.t_alpha = norm.ppf(1-alpha)
        
        # (iii) DGP for simulations
        assert mu > 0, 'Mean needs to be greater than zero'
        self.mu = mu
        assert (p > 0) and (p < 1), 'p needs to be between 0 and 1'
        self.p = p
        

    # Generate data from underlying parametric distribution
    """
    Generate scores from a Gaussian mixture N(mu,1) and N(0,1)
    """
    def sample(self, n, k=1, seed=None):
        # n=100;k=1;seed=1
        assert (n > 0) and isinstance(n, int), 'n needs to be an int > 0'
        if seed is not None:
            np.random.seed(seed)
        # Number of positive/negative samples
        y = np.random.binomial(n=1, p=self.p, size=[n, k])
        s = np.random.randn(n, k)
        s[y == 1] += self.mu
        return y, s

