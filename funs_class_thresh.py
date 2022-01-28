# Class needed to calibrate threshold for specific performance target
import numpy as np
from scipy.stats import norm
from funs_support import cvec
from funs_m import sensitivity

lst_m = ['sens','spec','prec']
lst_attr = ['learn_thresh', 'stat', 'oracle']


# Set up dictionary that can be called in
di_m = {'sens':sensitivity}

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
        self.m = di_m[m](mu=mu)  # Initialize performance measure class
        assert all([hasattr(self.m,attr) for attr in lst_attr])

        # (ii) Statistical testing
        assert (alpha > 0) and (alpha < 1), 'Type-I error must be between 0 to 1'
        self.alpha = alpha
        self.t_alpha = norm.ppf(1-alpha)
        
        # (iii) DGP for simulations
        assert mu > 0, 'Mean needs to be greater than zero'
        self.mu = mu
        assert (p > 0) and (p < 1), 'p needs to be between 0 and 1'
        self.p = p
        

    """
    Generate scores from a Gaussian mixture N(mu,1) and N(0,1)
    n:          Number of samples
    k:          Number of simulations (stored as columns)
    seed:       Seed for random draw
    keep:       Should scores and labels be stored as attributes?
    """
    def sample(self, n, k=1, seed=None, keep=False):
        # n=100;k=1;seed=1
        assert (n > 0) and isinstance(n, int), 'n needs to be an int > 0'
        if seed is not None:
            np.random.seed(seed)
        # Number of positive/negative samples
        self.y = np.random.binomial(n=1, p=self.p, size=[n, k])
        self.s = np.random.randn(n, k)
        self.s[self.y == 1] += self.mu


    """
    For a sample of data, learn the threshold
    y:          Binary labels
    s:          Scores
    gamma:      Target performance measure
    method:     Valid method for learning threshold (see lst_method)
    keep:       Should learned threshold be kept as attributes?
    """
    def learn_thresh(self, gamma, method, y=None, s=None):
        if y is None:
            y = self.y
        if s is None:
            s = self.s
        self.thresh = self.m.learn_thresh(y, s, gamma, method)
    
    """
    Oracle performance of learned thresholds
    thresh:     Vector of learned thresholds
    """
    def thresh2oracle(self, thresh=None):
        if thresh is None:
            thresh = self.thresh
        self.perf_oracle = self.m.oracle(thresh)
        
    """
    Get empirical performance measure
    """
    def thresh2emp(self, y=None, s=None, thresh=None):
        if y is None:
            y = self.y
        if s is None:
            s = self.s
        if thresh is None:
            thresh = self.thresh
        self.perf_emp = self.m.stat(y, s, thresh)


