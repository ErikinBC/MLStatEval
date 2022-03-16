# Class do power analysis on binomial proportion
import numpy as np
import pandas as pd
from scipy.stats import norm

class tools_power():
    """
    m:          Performance measure
    mu:         Mean of positive class N(mu, 1)
    p:          Probability of an observation being a positive class
    alpha:      Type-I error of the null hypothesis test
    """
    def __init__(self, m, mu=1, p=0.5, alpha=0.05):
        lst_m = ['sens','spec','prec']
        assert m in lst_m, 'm needs be one of: %s' % lst_m
        assert mu > 0, 'Mean needs to be greater than zero'
        self.mu = mu
        assert (p > 0) and (p < 1), 'p needs to be between 0 and 1'
        self.p = p
        # Set up statistical testing
        assert (alpha > 0) and (alpha < 1), 'Type-I error must be between 0 to 1'
        self.alpha = alpha
        self.t_alpha = norm.ppf(1-alpha)

    # Generate data from underlying parametric distribution
    """
    Generate scores from a Gaussian mixture N(mu,1) and N(0,1)
    """
    def dgp_bin(self, n, k=1, seed=None):
        assert (n > 0) and isinstance(n, int), 'n needs to be an int > 0'
        if seed is not None:
            np.random.seed(seed)
        # Number of positive/negative samples
        n1 = np.random.binomial(n=n, p=self.p)
        n0 = n - n1
        # Generate positive and negative scores
        s1 = self.mu + np.random.randn(n1)
        s0 = np.random.randn(n0)
        scores = np.append(s1, s0)
        labels = np.append(np.repeat(1, n1), np.repeat(0, n0))
        res = pd.DataFrame({'y':labels, 's':scores})
        return res
