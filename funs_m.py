# Performance measure functions (m)
import numpy as np
from scipy.stats import norm
from funs_support import cvec
from funs_stats import fast_quant_by_bool

# Quantile, Bootstrap-Quantile, Bootstrap-t, Bootstrap BCa
lst_method = ['quantile', 'bs-q', 'bs-t', 'bs-bca']



"""
Sensitivity (i.e. True Positive Rate)
"""
class sensitivity():
    # y=enc_thresh.y;s=enc_thresh.s;gamma=0.8;method='quantile'
    def __init__(self, mu):
        self.mu = mu

    def oracle(self, thresh):
        return norm.cdf(self.mu - thresh)

    @staticmethod
    def stat(y, s, t):
        yhat = np.where(s >= t, 1, 0)
        tps = np.sum((y == yhat) * (y == 1), 0)
        ps = y.sum(0)
        sens = tps / ps
        return sens

    @staticmethod
    def learn_thresh(y, s, gamma, method):
        assert method in lst_method
        assert (gamma >= 0) & (gamma <= 1)
        # Make scores into column vectors
        if len(y.shape) == 1:
            y, s = cvec(y), cvec(s)
        kk = y.shape[1]
        if method == 'quantile':
            thresh = fast_quant_by_bool(data=s, boolean=(y==1), q=1-gamma)
        return thresh
