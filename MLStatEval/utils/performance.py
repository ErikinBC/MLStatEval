# Performance measure functions (m)
# Each needs to have a oracle, stat, and learn_thresh method
# http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf

import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.stats import norm

# Internal packages
from MLStatEval.utils.utils import check01, get_cn_idx, clean_y_s, clean_y_s_threshold, clean_threshold
from MLStatEval.utils.vectorized import quant_by_col, quant_by_bool, loo_quant_by_bool

# List of valid methods for .learn_threshold
# point estimate, se(BS), Quantile, Bootstrap BCa
lst_method = ['point', 'basic', 'percentile', 'bca']

# self = sens_or_spec(choice=m, method=lst_method, alpha=0.05, n_bs=1000, seed=1)
class sens_or_spec():
    def __init__(self, choice, gamma, alpha=0.05):
        """
        Modular function for either sensitivity or specificity (avoids duplicated code)

        Inputs:
        choice:         String choice for either "sensitivity" or "specificity"
        gamma:          Performance measure target
        alpha:          Type-I error rate (for threshold inequality)
        """
        assert choice in ['sensitivity', 'specificity']
        self.choice = choice
        # Assign j label for use later
        self.j = 0
        if choice == 'sensitivity':
            self.j = 1
        assert check01(gamma), 'gamma needs to be between (0,1)'
        # Do we want to take the gamma or 1-gamma quantile of score distribution?
        self.m_gamma = gamma
        if self.choice == 'sensitivity':
            self.m_gamma = 1-gamma
        assert check01(alpha), 'alpha needs to be between (0,1)'
        self.alpha = alpha            

    def statistic(self, y, s, threshold):
        """
        Calculates sensitivity or specific4ity
        
        Inputs:
        y:                  Binary labels
        s:                  Scores
        threshold:          Operating threshold
        """
        # Clean up user input
        cn, idx, y, s, threshold = clean_y_s_threshold(y, s, threshold)
        # Calculate sensitivity or specificity
        yhat = np.where(s >= threshold, 1, 0)
        if self.choice == 'sensitivity':
            tps = np.sum((y == yhat) * (y == 1), axis=0)  # Intergrate out rows
            ps = np.sum(y, axis=0)
            score = tps / ps
        else:  # specificity
            tns = np.sum((y == yhat) * (y == 0), axis=0)  # Intergrate out rows
            ns = np.sum(1-y, axis=0)
            score = tns / ns
        if score.shape[1] == 1:
            score = score.flatten()
        if isinstance(cn, list):
            # If threshold was a DataFrame, retirn one as well
            score = pd.DataFrame(score, columns = cn, index=idx)
        return score

    """
    Different CI approaches for threshold for gamma target

    Inputs:
    y:              Binary labels
    s:              Scores
    n_bs:           # of bootstrap iterations
    seed:           Random seed
    method:         An inference method
    n_bs:           # of bootstrap iterations

    """
    def learn_threshold(self, y, s, method='percentile', n_bs=1000, seed=None):
        
        if isinstance(method, str):
            assert method in lst_method, 'method for learn_threshold must be one of: %s' % lst_method
            self.method = [method]
        else:
            assert all([meth in lst_method for meth in method]), 'method list must only contain valid methods: %s' % lst_method
            self.method = method
        assert n_bs > 0, 'number of bootstrap iterations must be positive!'
        self.n_bs = int(n_bs)
        if seed is not None:
            assert seed > 0, 'seed must be positive!'
            self.seed = int(seed)
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
        threshold = quant_by_bool(data=s, boolean=y_bool, q=self.m_gamma, interpolate='linear')
        y_bs = pd.DataFrame(y).sample(frac=self.n_bs, replace=True, random_state=self.seed)
        shape = (self.n_bs,)+y.shape
        y_bs_val = y_bs.values.reshape(shape)
        s_bs_val = pd.DataFrame(s).loc[y_bs.index].values.reshape(shape)
        y_bs_bool = (y_bs_val==self.j)
        threshold_bs = quant_by_bool(data=s_bs_val, boolean=y_bs_bool, q=self.m_gamma, interpolate='linear')
        # Return based on method
        di_threshold = dict.fromkeys(self.method)
        if 'point' in self.method:
            # i) "point": point esimate
            threshold_point = threshold.copy()
            di_threshold['point'] = threshold_point
        if 'basic' in self.method:
            # ii) "basic": point estimate ± standard error*quantile
            se_bs = threshold_bs.std(ddof=1,axis=0)
            threshold_basic = threshold + se_bs*q_alpha
            di_threshold['basic'] = threshold_basic
        if 'percentile' in self.method:
            # iii) "percentile": Use the alpha/1-alpha percentile of BS dist
            threshold_perc = np.quantile(threshold_bs, m_alpha, axis=0)
            di_threshold['percentile'] = threshold_perc
        if 'bca' in self.method:
            # iv) Bias-corrected and accelerated
            # Calculate LOO statistics
            threshold_loo = loo_quant_by_bool(data=s, boolean=y_bool, q=self.m_gamma)
            threshold_loo_mu = np.expand_dims(bn.nanmean(threshold_loo, axis=1), 1)
            # BCa calculation
            zhat0 = norm.ppf((np.sum(threshold_bs < threshold, 0) + 1) / (self.n_bs + 1))
            num = bn.nansum((threshold_loo_mu - threshold_loo)**3,axis=1)
            den = 6* (bn.nansum((threshold_loo_mu - threshold_loo)**2,axis=1))**(3/2)
            ahat = num / den
            alpha_adj = norm.cdf(zhat0 + (zhat0+z_alpha)/(1-ahat*(zhat0+z_alpha))).flatten()
            if self.choice == 'specificity':
                alpha_adj = 1 - alpha_adj
            threshold_bca = quant_by_col(threshold_bs, alpha_adj)
            di_threshold['bca'] = threshold_bca
        # Return different CIs
        res_ci = pd.DataFrame.from_dict(di_threshold)
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
  def __init__(self, gamma, alpha=0.05):
      sens_or_spec.__init__(self, choice='sensitivity', gamma=gamma, alpha=alpha)

# Wrapper for specificity
class specificity(sens_or_spec):
  def __init__(self, gamma, alpha=0.05):
      sens_or_spec.__init__(self, choice='specificity', gamma=gamma, alpha=alpha)


class precision():
    def statistic(self, y, s, thresh):
        """
        Calculates sensitivity or specificity
        y:          Binary labels
        s:          Scores
        thresh:     Operating threshold
        """
        cn, idx = get_cn_idx(thresh)
        thresh = clean_threshold(thresh)
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
