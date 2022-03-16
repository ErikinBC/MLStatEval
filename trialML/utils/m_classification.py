"""
Performance measure functions (m) for classification

For more details on bootstrapping methods, see http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf

A performance measure should have the following structure:

class m():
    def __init__(self, gamma, alpha):
        ...

    def statistic(self, y, s, threshold, return_den=False):
        ...

    def learn_threshold(self, y, s, method, n_bs, seed):
        ...

    def estimate_power(self, spread, n_trial):
        ...
"""

import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.stats import norm

# Internal packages
from trialML.utils.theory import power_binom
from trialML.utils.bootstrap import bca_calc
from trialML.utils.stats import umbrella_thresh
from trialML.utils.vectorized import quant_by_bool, loo_quant_by_bool, find_empirical_precision, loo_precision
from trialML.utils.utils import check01, clean_y_s, clean_y_s_threshold, array_to_float, try_flatten

"""
List of valid methods for .learn_threshold

point:                  point estimate
basic:                  Use the se(bs) to add on z_alpha deviations
percentile:             Use the alpha (or 1-alpha) percentile
bca:                    Bias-corrected and accelerated bootstrap
umbrella:               Neyman-Pearson Umbrella
"""
lst_method = ['point', 'basic', 'percentile', 'bca', 'umbrella']

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
        self.gamma = gamma
        self.m_gamma = gamma
        if self.choice == 'sensitivity':
            self.m_gamma = 1-gamma
        assert check01(alpha), 'alpha needs to be between (0,1)'
        self.alpha = alpha

    def estimate_power(self, spread, n_trial):
        """
        spread:             Null hypothesis spread (gamma - gamma_{H0})
        n_trial:            Expected number of trial points (note this is class specific!)
        """
        power = power_binom(spread, n_trial, self.gamma, self.alpha)
        return power

    def statistic(self, y, s, threshold, return_den=False):
        """
        Calculates sensitivity or specificity
        
        Inputs:
        y:                  Binary labels
        s:                  Scores
        threshold:          Operating threshold
        return_den:         Should the denominator of statistic be returned?
        """
        # Clean up user input
        cn, idx, y, s, threshold = clean_y_s_threshold(y, s, threshold)
        # Calculate sensitivity or specificity
        yhat = np.where(s >= threshold, 1, 0)
        if self.choice == 'sensitivity':
            tps = np.sum((y == yhat) * (y == 1), axis=0)  # Intergrate out rows
            den = np.sum(y, axis=0)  # Number of positives
            score = tps / den
        else:  # specificity
            tns = np.sum((y == yhat) * (y == 0), axis=0)  # Intergrate out rows
            den = np.sum(1-y, axis=0)  # Number of negatives
            score = tns / den
        nc_score = score.shape[1]
        nc_den = den.shape[1]
        # Flatten if possible
        score = try_flatten(score)
        if nc_score > nc_den == 1:
            # Duplicates columns
            den = np.tile(den, [1, nc_score])
        den = try_flatten(den)
        if isinstance(cn, list):
            # If threshold was a DataFrame, return one as well
            score = pd.DataFrame(score, columns = cn, index=idx)
            den = pd.DataFrame(den, columns = cn, index=idx)
        # Return as a float when relevant
        score = array_to_float(score)
        den = array_to_float(den)
        if return_den:                
            return score, den
        else:
            return score


    def learn_threshold(self, y, s, method='percentile', n_bs=1000, seed=None):
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
        if isinstance(method, str):
            assert method in lst_method, 'method for learn_threshold must be one of: %s' % lst_method
            self.method = [method]
        else:
            assert all([meth in lst_method for meth in method]), 'method list must only contain valid methods: %s' % lst_method
            self.method = method
        assert n_bs > 0, 'number of bootstrap iterations must be positive!'
        n_bs = int(n_bs)
        if seed is not None:
            assert seed > 0, 'seed must be positive!'
        # Do we want to add or subtract off z standard deviations?
        m_alpha = self.alpha
        if self.choice == 'specificity':
            m_alpha = 1 - self.alpha
        q_alpha = norm.ppf(m_alpha)
        upper = True
        if self.choice == 'sensitivity':
            upper = False
        # Make scores into column vectors
        y, s = clean_y_s(y, s)
        y_bool = (y==self.j)
        # Calculate point estimate
        threshold = quant_by_bool(data=s, boolean=y_bool, q=self.m_gamma, interpolate='linear')
        # Return based on method
        di_threshold = dict.fromkeys(self.method)
        if 'point' in self.method:
            # i) "point": point esimate
            threshold_point = threshold.copy()
            di_threshold['point'] = threshold_point
        if len(np.setdiff1d(self.method, ['point'])) > 0:
            # Run bootstrap
            y_bs = pd.DataFrame(y).sample(frac=n_bs, replace=True, random_state=seed)
            shape = (n_bs,)+y.shape
            y_bs_val = y_bs.values.reshape(shape)
            s_bs_val = pd.DataFrame(s).loc[y_bs.index].values.reshape(shape)
            y_bs_bool = (y_bs_val==self.j)
            threshold_bs = quant_by_bool(data=s_bs_val, boolean=y_bs_bool, q=self.m_gamma, interpolate='linear')
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
                threshold_bca = bca_calc(loo=np.squeeze(threshold_loo),bs=threshold_bs.T, baseline=threshold, alpha=self.alpha, axis=0,upper=upper)
                di_threshold['bca'] = threshold_bca
            if 'umbrella' in self.method:
                if self.choice == 'sensitivity':
                    n = np.sum(y, axis=0)
                    s_sort = np.sort(np.where(y == 1, s, np.nan), axis=0)
                else:
                    n = np.sum(1 - y, axis=0)
                    s_sort = np.sort(np.where(y == 0, s, np.nan), axis=0)
                idx_umb = umbrella_thresh(n=n ,target=self.gamma, alpha=self.alpha, upper=upper)
                threshold_umb = np.take_along_axis(s_sort, idx_umb[None], axis=0)
                threshold_umb = try_flatten(threshold_umb)
                di_threshold['umbrella'] = threshold_umb
        # Return different CIs
        res_ci = pd.DataFrame.from_dict(di_threshold)
        # If it's a 1x1 array or dataframe, return as a float
        res_ci = array_to_float(res_ci)
        return res_ci


# Wrapper for sensitivity
class sensitivity(sens_or_spec):
  def __init__(self, gamma, alpha=0.05):
      sens_or_spec.__init__(self, choice='sensitivity', gamma=gamma, alpha=alpha)

# Wrapper for specificity
class specificity(sens_or_spec):
  def __init__(self, gamma, alpha=0.05):
      sens_or_spec.__init__(self, choice='specificity', gamma=gamma, alpha=alpha)


# from trialML.theory import gaussian_mixture
# normal_dgp = gaussian_mixture()
# normal_dgp.set_params(0.5,1,0,1,1)
# y, s = normal_dgp.gen_mixture(100,20, seed=1)
# gamma=0.6;alpha=0.05
# normal_dgp.set_gamma(gamma)
# self=precision(gamma=gamma,alpha=alpha)
class precision():
    def __init__(self, gamma, alpha=0.05):
        assert check01(gamma), 'gamma needs to be between (0,1)'
        self.gamma = gamma
        self.alpha = alpha
        assert check01(alpha), 'alpha needs to be between (0,1)'
        self.gamma = gamma
        self.alpha = alpha

    # threshold=0.2;return_den=True
    @staticmethod
    def statistic(y, s, threshold, return_den=False):
        """Calculates the precision

        Inputs:
        y:                  Binary labels
        s:                  Scores
        threshold:          Operating threshold
        return_den:         Should the denominator of statistic be returned?
        """
        # Clean up user input
        cn, idx, y, s, threshold = clean_y_s_threshold(y, s, threshold)
        # Predicted positives and precision
        yhat = np.where(s >= threshold, 1, 0)
        tps = np.sum((yhat == 1) * (y == 1), axis=0)  # Intergrate out rows
        den = np.sum(yhat, axis=0)
        score = tps / den  # PPV
        nc_score = score.shape[1]
        nc_den = den.shape[1]
        score = try_flatten(score)
        den = try_flatten(den)
        if nc_score > nc_den == 1:
            # Duplicates columns
            den = np.tile(den, [1, nc_score])
        if isinstance(cn, list):
            # If threshold was a DataFrame, return one as well
            score = pd.DataFrame(score, columns = cn, index=idx)
            den = pd.DataFrame(den, columns = cn, index=idx)
        # Return as a float when relevant
        score = array_to_float(score)
        den = array_to_float(den)
        if return_den:                
            return score, den
        else:
            return score
        
    # method=lst_method;n_bs=1000;seed=1
    def learn_threshold(self, y, s, method='percentile', n_bs=1000, seed=None):
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
        if isinstance(method, str):
            assert method in lst_method, 'method for learn_threshold must be one of: %s' % lst_method
            self.method = [method]
        else:
            assert all([meth in lst_method for meth in method]), 'method list must only contain valid methods: %s' % lst_method
            self.method = method
        assert n_bs > 0, 'number of bootstrap iterations must be positive!'
        n_bs = int(n_bs)
        if seed is not None:
            assert seed > 0, 'seed must be positive!'
        # We use 1-alpha since we want to pick upper bound
        m_alpha = 1 - self.alpha
        z_alpha = norm.ppf(m_alpha)
        # Make scores into column vectors
        y, s = clean_y_s(y, s)
        # Calculate point estimate and bootstrap
        threshold = find_empirical_precision(y=y, s=s, target=self.gamma)
        # Return based on method
        di_threshold = dict.fromkeys(self.method)
        if 'point' in self.method:  # i) "point": point esimate
            threshold_point = threshold.copy()
            di_threshold['point'] = threshold_point
        # Generate bootstrap distribution
        if len(np.setdiff1d(self.method, ['point'])) > 0:    
            y_bs = pd.DataFrame(y).sample(frac=n_bs, replace=True, random_state=seed)
            shape = (n_bs,)+y.shape
            y_bs_val = y_bs.values.reshape(shape)
            s_bs_val = pd.DataFrame(s).loc[y_bs.index].values.reshape(shape)
            # precision function needs axis order to be (# of observations) x (# of columns) x (# of simulations)
            tidx = [1,2,0]
            y_bs_val = y_bs_val.transpose(tidx)
            s_bs_val = s_bs_val.transpose(tidx)
            # Recalculate precision threshold on bootstrapped data
            threshold_bs = find_empirical_precision(y=y_bs_val, s=s_bs_val, target=self.gamma)
            if 'basic' in self.method:  # ii) "basic": point estimate ± standard error*quantile
                se_bs = bn.nanstd(threshold_bs, ddof=1, axis=1)
                threshold_basic = threshold + se_bs*z_alpha
                di_threshold['basic'] = threshold_basic
            if 'percentile' in self.method:  # iii) "percentile": Use the alpha/1-alpha percentile of BS dist
                thresh_bool = ~np.isnan(threshold_bs)  # Some values are nan if threshold value cannot be obtained
                # Transpose applied to match dimension structure of sens/spec
                threshold_perc = quant_by_bool(threshold_bs.T, thresh_bool.T, m_alpha)
                di_threshold['percentile'] = threshold_perc
            if 'bca' in self.method:  # iv) Bias-corrected and accelerated
                # Calculate LOO statistics
                threshold_loo = loo_precision(y, s, self.gamma)
                threshold_bca = bca_calc(loo=threshold_loo, bs=threshold_bs, baseline=threshold, alpha=self.alpha, upper=True)
                di_threshold['bca'] = threshold_bca
        # Return different CIs
        res_ci = pd.DataFrame.from_dict(di_threshold)
        # Remove null columns
        res_ci = res_ci.loc[:,~res_ci.isnull().all()]
        # If it's a 1x1 array or dataframe, return as a float
        res_ci = array_to_float(res_ci)
        return res_ci        

    def estimate_power(self, spread, n_trial):
        """
        spread:             Null hypothesis spread (gamma - gamma_{H0})
        n_trial:            Expected number of trial points (note this is class specific!)
        """
        power = power_binom(spread, n_trial, self.gamma, self.alpha)
        return power
