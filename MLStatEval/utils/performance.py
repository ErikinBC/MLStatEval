# Performance measure functions (m)
# Each needs to have a oracle, stat, and learn_thresh method
# http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf

from glob import glob
import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.stats import norm
from utils import cvec, get_cn_idx, clean_thresh
from funs_vectorized import quant_by_col, quant_by_bool, loo_quant_by_bool

class base_m():
    """
    Base class used for different performance measures
    alpha:      Type-I error rate
    mu:         Mean of positive class normal dist
    p:          P(y==1) - Not relevant for sens/spec
    """
    def __init__(self, alpha=0.05, mu=1, p=None):
        assert alpha < 0.5, 'alpha must be less than 50%'
        self.alpha = alpha
        self.mu = mu
        assert (p > 0) & (p < 1), 'p must be between 0 and 1'
        self.p = p


"""
Modular function for either sensitivity or specificity
"""
class sens_or_spec(base_m):
    def __init__(self, alpha=0.05, mu=1, p=None):
      sens_or_spec.__init__(self, m='sens')

    def __init__(self, m):
        """
        m:          Performance measure function (either "sens" or "spec")
        """
        assert m in ['sens','spec'], 'set m to either "sens" or "spec"'
        self.m = m

    """
    Maps an operating threshold to an oracle performance measure
    """
    def oracle(self, thresh):
        cn, idx = get_cn_idx(thresh)
        if self.m == 'sens':
            z = norm.cdf(self.mu - thresh)
        else:
            z = norm.cdf(thresh)
        if isinstance(cn, list):
            z = pd.DataFrame(z, columns=cn, index=idx)
        return z

    """
    Calculates sensitivity or specificity
    y:          Binary labels
    s:          Scores
    thresh:     Operating threshold
    """
    def stat(self, y, s, thresh):
        cn, idx = get_cn_idx(thresh)
        thresh = clean_thresh(thresh)
        # Ensure operations broadcast
        y_shape = y.shape + (1,)
        t_shape = (1,) + thresh.shape
        y = y.reshape(y_shape)
        s = s.reshape(y_shape)
        thresh = thresh.reshape(t_shape)
        assert len(s.shape) == len(thresh.shape)
        # Calculate sensitivity or specificity
        yhat = np.where(s >= thresh, 1, 0)
        if self.m == 'sens':
            tps = np.sum((y == yhat) * (y == 1), axis=0)  # Intergrate out rows
            ps = np.sum(y, axis=0)
            score = tps / ps
        else:
            tns = np.sum((y == yhat) * (y == 0), axis=0)  # Intergrate out rows
            ns = np.sum(1-y, axis=0)
            score = tns / ns
        if score.shape[1] == 1:
            score = score.flatten()
        if isinstance(cn, list):
            score = pd.DataFrame(score, columns = cn, index=idx)
        return score

    """
    Different CI approaches for threshold for gamma target
    n_bs:       # of bootstrap iterations
    seed:       Random seed
    """
    def learn_thresh(self, y, s, gamma, n_bs=1000, seed=None):
        assert (gamma >= 0) & (gamma <= 1)
        # Oracle threshold based on gamma
        if self.m == 'sens':
            m_gamma = 1-gamma
            self.thresh_gamma = self.mu + norm.ppf(m_gamma)
            y = cvec(y.copy())
            alpha = self.alpha
        else:
            m_gamma = gamma
            self.thresh_gamma = norm.ppf(gamma)
            y = cvec(1 - y.copy())
            alpha = 1 - self.alpha
        # Make scores into column vectors
        s = cvec(s.copy())
        # (i) Calculate point esimate
        thresh = quant_by_bool(data=s, boolean=(y==1), q=m_gamma, interpolate='linear')
        # (ii) Calculate bootstrap range
        y_bs = pd.DataFrame(y).sample(frac=n_bs, replace=True, random_state=seed)
        shape = (n_bs,)+y.shape
        y_bs_val = y_bs.values.reshape(shape)
        s_bs_val = pd.DataFrame(s).loc[y_bs.index].values.reshape(shape)
        thresh_bs = quant_by_bool(data=s_bs_val, boolean=(y_bs_val==1), q=m_gamma, interpolate='linear')
        # (iii) Calculate LOO statistics
        thresh_loo = loo_quant_by_bool(data=s, boolean=(y==1), q=m_gamma)
        thresh_loo_mu = np.expand_dims(bn.nanmean(thresh_loo, axis=1), 1)
        # (iv) Basic CI approaches
        z_alpha = norm.ppf(alpha)
        ci_quantile = np.quantile(thresh_bs, alpha, axis=0)
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

# Wrapper for sensitivity
class sensitivity(sens_or_spec):
  def __init__(self, alpha=0.05, mu=1, p=None):
      sens_or_spec.__init__(self, m='sens', alpha=alpha, mu=mu, p=p)
    
# Wrapper for specificity
class specificity(sens_or_spec):
  def __init__(self, alpha=0.05, mu=1, p=None):
      sens_or_spec.__init__(self, m='spec', alpha=alpha, mu=mu, p=p)
    




# # Quantile, Bootstrap-Quantile, Bootstrap-t, Bootstrap BCa
# lst_method = ['quantile', 'bs-q', 'bs-t', 'bs-bca']

class precision(base_m):

    def oracle(self, thresh):
        """
        Maps an operating threshold to an oracle precision
        """
        cn, idx = get_cn_idx(thresh)
        # TPR
        term1 = norm.cdf(self.mu - thresh)*self.p
        # FPR
        term2 = norm.cdf(-thresh) * (1-self.p)
        z = term1 / (term1 + term2)
        if isinstance(cn, list):
            z = pd.DataFrame(z, columns=cn, index=idx)
        return z

    def stat(self, y, s, thresh):
        """
        Calculates sensitivity or specificity
        y:          Binary labels
        s:          Scores
        thresh:     Operating threshold
        """
        cn, idx = get_cn_idx(thresh)
        thresh = clean_thresh(thresh)
        # (i) Ensure operations broadcast
        y_shape = y.shape + (1,)
        t_shape = (1,) + thresh.shape
        y = y.reshape(y_shape)
        s = s.reshape(y_shape)
        thresh = thresh.reshape(t_shape)
        assert len(s.shape) == len(thresh.shape)
        # (ii) Calculate metric
        yhat = np.where(s >= thresh, 1, 0)
        tp = np.sum((y == yhat) * (y == 1), axis=0)  # Intergrate out rows
        p = np.sum(yhat, axis=0)
        score = tp / p
        if score.shape[1] == 1:
            score = score.flatten()
        if isinstance(cn, list):
            score = pd.DataFrame(score, columns = cn, index=idx)
        return score

    def learn_thresh(self, y, s, gamma, n_bs=1000, seed=None):
        assert (gamma >= 0) & (gamma <= 1)
        # Oracle threshold based on gamma
        """
        Different CI approaches for threshold for gamma target
        y:          Binary labels
        s:          Scores
        gamma:      Target sens or spec
        n_bs:       # of bootstrap iterations
        seed:       Random seed
        """


# thresh, p, n, k = 1, 0.25, 100, 250
# self = precision(alpha=0.05,mu=1,p=p)
# y=np.where(np.random.rand(n,k)>1-p,1,0);s=np.random.randn(n,k)
# self.oracle(thresh)
# self.stat(y, s, thresh).mean()


