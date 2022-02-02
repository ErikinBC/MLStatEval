# Performance measure functions (m)
# Each needs to have a oracle, stat, and learn_thresh method
# http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf

import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.stats import norm
from funs_support import cvec
from funs_vectorized import quant_by_col, quant_by_bool, loo_quant_by_bool

# Quantile, Bootstrap-Quantile, Bootstrap-t, Bootstrap BCa
lst_method = ['quantile', 'bs-q', 'bs-t', 'bs-bca']

"""
Sensitivity (i.e. True Positive Rate)
"""
class sensitivity():
    # y=enc_thresh.y;s=enc_thresh.s;gamma=0.8;method='quantile'
    def __init__(self, alpha, mu, p=None):
        self.alpha = alpha
        self.mu = mu
        self.interpolate = 'linear'  # Only use linear-quantile method for now

    def oracle(self, thresh):
        cn = None
        if isinstance(thresh, pd.DataFrame):
            cn = list(thresh.columns)
            idx = thresh.index
        z = norm.cdf(self.mu - thresh)
        if isinstance(cn, list):
            z = pd.DataFrame(z, columns=cn, index=idx)
        return z

    @staticmethod
    def stat(y, s, t):
        # y=enc_thresh.y;s=enc_thresh.s;t=0.15#enc_thresh.thresh.point
        cn = None
        if isinstance(t, pd.DataFrame):
            cn = list(t.columns)
            idx = t.index
        if isinstance(t, float) or isinstance(t, int):
            t = np.array([t])
        if not isinstance(t, np.ndarray):
            t = np.array(t)
        t = cvec(t)
        # Ensure operations broadcast
        y_shape = y.shape + (1,)
        t_shape = (1,) + t.shape
        y = y.reshape(y_shape)
        s = s.reshape(y_shape)
        t = t.reshape(t_shape)
        assert len(s.shape) == len(t.shape)
        # Calculate TPs relative to positives
        yhat = np.where(s >= t, 1, 0)
        tps = np.sum((y == yhat) * (y == 1), axis=0)  # Intergrate out rows
        ps = np.sum(y, axis=0)
        sens = tps / ps
        if sens.shape[1] == 1:
            sens = sens.flatten()
        if isinstance(cn, list):
            sens = pd.DataFrame(sens, columns = cn, index=idx)
        return sens

    # self=enc_thresh.m['sens']; y=enc_thresh.y;s=enc_thresh.s;n_bs=200;seed=1
    def learn_thresh(self, y, s, gamma, n_bs=1000, seed=None):
        # assert method in lst_method
        assert (gamma >= 0) & (gamma <= 1)
        # Oracle threshold based on gamma
        self.thresh_gamma = self.mu + norm.ppf(1-gamma)
        # Make scores into column vectors
        y = cvec(y.copy())
        s = cvec(s.copy())
        # (i) Calculate point esimate
        thresh = quant_by_bool(data=s, boolean=(y==1), q=1-gamma, interpolate=self.interpolate)
        # (ii) Calculate bootstrap range
        y_bs = pd.DataFrame(y).sample(frac=n_bs, replace=True, random_state=seed)
        shape = (n_bs,)+y.shape
        y_bs_val = y_bs.values.reshape(shape)
        s_bs_val = pd.DataFrame(s).loc[y_bs.index].values.reshape(shape)
        thresh_bs = quant_by_bool(data=s_bs_val, boolean=(y_bs_val==1), q=1-gamma, interpolate=self.interpolate)
        # (iii) Calculate LOO statistics
        thresh_loo = loo_quant_by_bool(data=s, boolean=(y==1), q=1-gamma)
        thresh_loo_mu = np.expand_dims(bn.nanmean(thresh_loo, axis=1), 1)
        # (iv) Basic CI approaches
        z_alpha = norm.ppf(self.alpha)
        ci_quantile = np.quantile(thresh_bs, self.alpha, axis=0)
        se_bs = thresh_bs.std(ddof=1,axis=0)
        ci_basic = thresh + se_bs*z_alpha
        # (v) BCa calculation
        zhat0 = norm.ppf((np.sum(thresh_bs < thresh, 0) + 1) / (n_bs + 1))
        num = bn.nansum((thresh_loo_mu - thresh_loo)**3,axis=1)
        den = 6* (bn.nansum((thresh_loo_mu - thresh_loo)**2,axis=1))**(3/2)
        ahat = num / den
        alpha_adj = norm.cdf(zhat0 + (zhat0+z_alpha)/(1-ahat*(zhat0+z_alpha))).flatten()
        ci_bca = quant_by_col(thresh_bs, alpha_adj)

        # Return different CIs
        res_ci = pd.DataFrame({'point':thresh, 'basic':ci_basic, 'quantile':ci_quantile, 'bca':ci_bca})
        return res_ci

