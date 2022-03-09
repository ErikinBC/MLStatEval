# Performance measure functions (m)
# Each needs to have a oracle, stat, and learn_thresh method
# http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf

import sys
import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.stats import norm

# Internal packages
from MLStatEval.utils.utils import check01, cvec, get_cn_idx, clean_y_s, clean_y_s_threshold
from MLStatEval.utils.vectorized import quant_by_col, quant_by_bool, loo_quant_by_bool


# List of valid methods for .learn_threshold
# point estimate, se(BS), Quantile, Bootstrap BCa
lst_method = ['point', 'basic', 'percentile', 'bca']

# self = sens_or_spec(choice='sensitivity', method='percentile', alpha=0.05, n_bs=1000, seed=None)
class sens_or_spec():
    def __init__(self, choice, method='percentile', alpha=0.05, n_bs=1000, seed=None):    
        """
        Modular function for either sensitivity or specificity (avoids duplicated code)

        choice:         String choice for either "sensitivity" or "specificity"
        alpha:          Type-I error rate (for threshold inequality)
        """
        assert choice in ['sensitivity', 'specificity']
        self.choice = choice
        # Assign j label for use later
        self.j = 0
        if choice == 'sensitivity':
            self.j = 1
        if isinstance(method, str):
            assert method in lst_method, 'method for learn_threshold must be one of: %s' % lst_method
            self.method = [method]
        else:
            assert all([meth in lst_method for meth in method]), 'method list must only contain valid methods: %s' % lst_method
        assert check01(alpha), 'alpha needs to be between (0,1)'
        self.alpha = alpha
        assert n_bs > 0, 'number of bootstrap iterations must be positive!'
        self.n_bs = int(n_bs)
        if seed is not None:
            assert seed > 0, 'seed must be positive!'
            self.seed = int(seed)
        else:
            self.seed = None
            

    def statistic(self, y, s, threshold):
        """
        Calculates sensitivity or specificity
        y:                  Binary labels
        s:                  Scores
        threshold:          Operating threshold
        """
        # Clean up user input
        cn, idx, y, s, threshold = clean_y_s_threshold(y, s, threshold)
        # Calculate sensitivity or specificity
        yhat = np.where(s >= threshold, 1, 0)
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
    # NEED TO CHECK FOR BOTH SENS & SPEC!!!
    def learn_threshold(self, y, s, gamma):
        assert check01(gamma), 'gamma needs to be between (0,1)'
        # Do we want to take the gamma or 1-gamma quantile of score distribution?
        m_gamma = gamma
        if self.choice == 'sensitivity':
            m_gamma = 1-gamma
        # Do we want to add or subtract off z standard deviations?
        z_alpha = norm.ppf(self.alpha)
        m_alpha = self.alpha
        if self.choice == 'specificity':
            m_alpha = 1 - self.alpha
        q_alpha = norm.ppf(m_alpha)
        # Make scores into column vectors
        y, s = clean_y_s(y, s)
        y_bool = (y==self.j)
        # Calculate point estimate and bootstrap
        thresh = quant_by_bool(data=s, boolean=y_bool, q=m_gamma, interpolate='linear')
        y_bs = pd.DataFrame(y).sample(frac=self.n_bs, replace=True, random_state=self.seed)
        shape = (self.n_bs,)+y.shape
        y_bs_val = y_bs.values.reshape(shape)
        s_bs_val = pd.DataFrame(s).loc[y_bs.index].values.reshape(shape)
        y_bs_bool = (y_bs_val==self.j)
        thresh_bs = quant_by_bool(data=s_bs_val, boolean=y_bs_bool, q=m_gamma, interpolate='linear')
        # Return based on method
        di_thresh = dict.fromkeys(self.method)
        if 'point' in self.method:
            # i) "point": point esimate
            thresh_point = thresh
            di_thresh['point'] = thresh_point
        elif 'basic' in self.method:
            # ii) "basic": point estimate ± standard error*quantile
            se_bs = thresh_bs.std(ddof=1,axis=0)
            thresh_basic = thresh + se_bs*q_alpha
            di_thresh['basic'] = thresh_basic
        elif 'percentile' in self.method:
            # iii) "percentile": Use the alpha/1-alpha percentile of BS dist
            thresh_perc = np.quantile(thresh_bs, m_alpha, axis=0)
            di_thresh['percentile'] = thresh_perc
        elif 'bca' in self.method:
            # iv) Bias-corrected and accelerated
            # Calculate LOO statistics
            thresh_loo = loo_quant_by_bool(data=s, boolean=y_bool, q=m_gamma)
            thresh_loo_mu = np.expand_dims(bn.nanmean(thresh_loo, axis=1), 1)
            # BCa calculation
            zhat0 = norm.ppf((np.sum(thresh_bs < thresh, 0) + 1) / (self.n_bs + 1))
            num = bn.nansum((thresh_loo_mu - thresh_loo)**3,axis=1)
            den = 6* (bn.nansum((thresh_loo_mu - thresh_loo)**2,axis=1))**(3/2)
            ahat = num / den
            alpha_adj = norm.cdf(zhat0 + (zhat0+z_alpha)/(1-ahat*(zhat0+z_alpha))).flatten()
            thresh_bca = quant_by_col(thresh_bs, alpha_adj)
            di_thresh['bca'] = thresh_bca
        else:
            sys.exit('How did we get here?!')
        # Return different CIs
        res_ci = pd.DataFrame.from_dict(di_thresh)
        return res_ci


# """
# Maps an operating threshold to an oracle performance measure
# """
# def oracle(self, thresh):
#     cn, idx = get_cn_idx(thresh)
#     if self.m == 'sens':
#         z = norm.cdf(self.mu - thresh)
#     else:
#         z = norm.cdf(thresh)
#     if isinstance(cn, list):
#         z = pd.DataFrame(z, columns=cn, index=idx)
#     return z


# Wrapper for sensitivity
class sensitivity(sens_or_spec):
  def __init__(self):
      sens_or_spec.__init__(self, choice='sensitivity')


# Wrapper for specificity
class specificity(sens_or_spec):
  def __init__(self, alpha=0.05, mu=1, p=None):
      sens_or_spec.__init__(self, m='spec', alpha=alpha, mu=mu, p=p)


class precision():
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


