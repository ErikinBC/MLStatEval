# Class needed to calibrate threshold for specific performance target
import numpy as np
import pandas as pd
from scipy.stats import norm
from funs_m import sensitivity
from funs_support import no_diff

lst_attr = ['learn_thresh', 'stat', 'oracle']

# Set up dictionary that can be called in
di_m = {'sens':sensitivity, 'spec':sensitivity, 'prec':sensitivity}

class tools_thresh():
    """
    m:          Performance measure
    mu:         Mean of positive class N(mu, 1)
    p:          Probability of an observation being a positive class
    alpha:      Type-I error of the null hypothesis test
    """
    def __init__(self, *args, alpha=0.05, mu=1, p=0.5):
        # (i) Performance measure
        # print(args)
        assert len(args) >= 1, 'need at least one performance measures in kwargs'
        assert all([m in di_m for m in args]), 'All kwargs need to be in lst_m'
        self.m = {k:None  for k in args}
        for k in args:
            self.m[k] = di_m[k](alpha=alpha, mu=mu)  # Initialize performance measure class
            assert all([hasattr(self.m[k],attr) for attr in lst_attr]), 'Performance measure %s needs the following attributes: %s' % (k, lst_attr)

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
    def learn_thresh(self, gamma, y=None, s=None):
        # y=None;s=None
        if y is None:
            y = self.y
        if s is None:
            s = self.s
        self.thresh = dict.fromkeys(self.m)
        for k in self.m:
            tmp_thresh = self.m[k].learn_thresh(y, s, gamma)
            tmp_thresh = tmp_thresh.rename_axis('cidx').reset_index()
            tmp_thresh = tmp_thresh.assign(m=k).set_index(['m', 'cidx'])
            self.thresh[k] = tmp_thresh
    
    """
    Oracle performance of learned thresholds
    thresh:     Vector of learned thresholds
    """
    def thresh2oracle(self, thresh=None):
        if thresh is None:
            thresh = self.thresh
        is_dict = isinstance(thresh, dict)
        self.perf_oracle = dict.fromkeys(self.m)
        for k in self.m:
            if is_dict:
                self.perf_oracle[k] = self.m[k].oracle(thresh[k])
            else:
                self.perf_oracle[k] = self.m[k].oracle(thresh)

        
    """
    Get empirical performance measure
    """
    def thresh2emp(self, y=None, s=None, thresh=None):
        # y=None;s=None;thresh=None
        if y is None:
            y = self.y
        if s is None:
            s = self.s
        if thresh is None:
            thresh = self.thresh
        is_dict = isinstance(thresh, dict)
        self.perf_emp = dict.fromkeys(self.m)
        for k in self.m:
            if is_dict:
                self.perf_emp[k] = self.m[k].stat(y, s, thresh[k])
            else:
                self.perf_emp[k] = self.m[k].oracstatle(y, s, thresh)

    """
    Merge dicts into single DataFrame
    """
    def merge_dicts(self):
        # (i) Merge thresholds/oracle/empirical together
        res_emp = pd.concat([df.reset_index() for df in self.perf_emp.values()])
        res_emp = res_emp.melt(['m','cidx'],None,'method','emp')
        res_oracle = pd.concat([df.reset_index() for df in self.perf_oracle.values()])
        res_oracle = res_oracle.melt(['m','cidx'],None,'method','oracle')
        res_thresh = pd.concat([df.reset_index() for df in self.thresh.values()])
        res_thresh = res_thresh.melt(['m','cidx'],None,'method','thresh')
        self.df_res = res_thresh.merge(res_emp).merge(res_oracle)
        # (ii) Merge the oracle thresholds
        self.thresh_gamma = pd.concat([pd.DataFrame({'m':k, 'thresh_gamma':v.thresh_gamma},index=[0]) for k,v in self.m.items()])
        self.thresh_gamma.reset_index(drop=True, inplace=True)