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
from MLStatEval.utils.bootstrap import bca_calc
from MLStatEval.utils.vectorized import quant_by_col, quant_by_bool, loo_quant_by_bool, find_empirical_precision, loo_precision
from MLStatEval.utils.utils import check01, df_cn_idx_args, clean_y_s, clean_y_s_threshold, to_array, array_to_float, try_flatten

"""
List of valid methods for .learn_threshold

point:                  point estimate
basic:                  Use the se(bs) to add on z_alpha deviations
percentile:             Use the alpha (or 1-alpha) percentile
bca:                    Bias-corrected and accelerated bootstrap
umbrella:               Neyman-Pearson Umbrella
"""
lst_method = ['point', 'basic', 'percentile', 'bca']  #, 'umbrella'

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
        cn, idx = df_cn_idx_args(spread, n_trial)
        # Allow for vectorization
        spread, n_trial = to_array(spread), to_array(n_trial)
        assert np.all(spread > 0) & np.all(spread < self.gamma), 'spread must be between (0, gamma)'
        gamma0 = self.gamma - spread
        sig0 = np.sqrt( gamma0*(1-gamma0) / n_trial )
        sig = np.sqrt( self.gamma*(1-self.gamma) / n_trial )
        z_alpha = norm.ppf(1-self.alpha)
        power = norm.cdf( (spread - sig0*z_alpha) / sig )
        power = array_to_float(power)
        if isinstance(cn, list):
            power = pd.DataFrame(power, columns = cn, index=idx)
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
            alpha_adj = norm.cdf(zhat0 + (zhat0+z_alpha)/(1-ahat*(zhat0+z_alpha)))
            alpha_adj = try_flatten(alpha_adj)
            if self.choice == 'specificity':
                alpha_adj = 1 - alpha_adj
            threshold_bca = quant_by_col(threshold_bs, alpha_adj)
            di_threshold['bca'] = threshold_bca
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


# from MLStatEval.theory import gaussian_mixture
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
        self.n_bs = int(n_bs)
        if seed is not None:
            assert seed > 0, 'seed must be positive!'
            self.seed = int(seed)
        # We use 1-alpha since we want to pick upper bound
        m_alpha = 1 - self.alpha
        z_alpha = norm.ppf(m_alpha)
        # Make scores into column vectors
        y, s = clean_y_s(y, s)
        # Calculate point estimate and bootstrap
        threshold = find_empirical_precision(y=y, s=s, target=self.gamma)
        # Generate bootstrap distribution
        y_bs = pd.DataFrame(y).sample(frac=self.n_bs, replace=True, random_state=self.seed)
        shape = (self.n_bs,)+y.shape
        y_bs_val = y_bs.values.reshape(shape)
        s_bs_val = pd.DataFrame(s).loc[y_bs.index].values.reshape(shape)
        # precision function needs axis order to be (# of observations) x (# of columns) x (# of simulations)
        tidx = [1,2,0]
        y_bs_val = y_bs_val.transpose(tidx)
        s_bs_val = s_bs_val.transpose(tidx)
        # Recalculate precision threshold on bootstrapped data
        threshold_bs = find_empirical_precision(y=y_bs_val, s=s_bs_val, target=self.gamma)
        # Return based on method
        di_threshold = dict.fromkeys(self.method)
        if 'point' in self.method:  # i) "point": point esimate
            threshold_point = threshold.copy()
            di_threshold['point'] = threshold_point
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
        # If it's a 1x1 array or dataframe, return as a float
        res_ci = array_to_float(res_ci)
        return res_ci        

    def estimate_power(self, spread, n_trial):
        """
        spread:             Null hypothesis spread (gamma - gamma_{H0})
        n_trial:            Expected number of trial points (note this is class specific!)
        """
        cn, idx = df_cn_idx_args(spread, n_trial)
        # Allow for vectorization
        spread, n_trial = to_array(spread), to_array(n_trial)
        assert np.all(spread > 0) & np.all(spread < self.gamma), 'spread must be between (0, gamma)'
        gamma0 = self.gamma - spread
        sig0 = np.sqrt( gamma0*(1-gamma0) / n_trial )
        sig = np.sqrt( self.gamma*(1-self.gamma) / n_trial )
        z_alpha = norm.ppf(1-self.alpha)
        power = norm.cdf( (spread - sig0*z_alpha) / sig )
        power = array_to_float(power)
        if isinstance(cn, list):
            power = pd.DataFrame(power, columns = cn, index=idx)
        return power
