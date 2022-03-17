"""
Classes to support two-sided power calculations
"""

# i) Assign two performance functions / alpha
# ii) Set y, s, and target performance for each
# iii) Power simulations (spread1, spread2, n1, n2)

import numpy as np
import pandas as pd
from scipy.stats import norm

# Internal modules
from trialML.utils.stats import get_CI
from trialML.utils.theory import power_binom
from trialML.utils.utils import check01, check_binary, to_array
from trialML.utils.m_classification import sensitivity, specificity, precision

di_m_classification = {'sensitivity':sensitivity, 'specificity':specificity, 'precision':precision}
lst_m = list(di_m_classification)

class twosided_classification():
    def __init__(self, m1, m2, alpha):
        """
        Provides 95% CIs for power for a classification performance measure with a binomial proportion. 

        Parameters
        ----------
        m1:         First performance measure
        m2:         Second performance measure
        alpah:      Type-I error rate for test statistic
        """
        assert check01(alpha), 'alpha must be between (0,1)'
        assert m1 in lst_m, 'm1 must be one of: %s' % lst_m
        assert m2 in lst_m, 'm2 must be one of: %s' % lst_m
        self.m1 = di_m_classification[m1]
        self.m2 = di_m_classification[m2]
        self.alpha = alpha

    def statistic_CI(self, y, s, threshold):
        """
        Calculates m1/m2 statistic for a given threshold and corresponding 95% CI

        Returns
        -------
        A DataFrame with an (m) index, the statistic (gamma_hat) with its (1-alpha)% CI, the ratio of test statistic observations to the overall sample size (ratio), and a column identify for s (cidx)
        """
        n = len(y)
        assert y.shape == s.shape, 'y and s need to have the same shape'
        assert check_binary(y), 'y needs to be [0,1]'
        # Get test statistic for each
        stat1, den1 = self.m1.statistic(y=y, s=s, threshold=threshold, return_den=True)
        stat2, den2 = self.m2.statistic(y=y, s=s, threshold=threshold, return_den=True)
        # Get the lb/ub ranges for the binomial confidence interval
        if isinstance(stat1, float):
            df1 = pd.DataFrame({'m':1, 'gamma_hat':stat1, 'den':den1},index=[0])
            df2 = pd.DataFrame({'m':2, 'gamma_hat':stat2, 'den':den2},index=[1])
        else:
            df1 = pd.DataFrame({'m':1, 'gamma_hat':stat1.values.flat, 'den':den1.values.flat})
            df2 = pd.DataFrame({'m':2,'gamma_hat':stat2.values.flat, 'den':den2.values.flat})
        df = pd.concat(objs=[df1, df2], axis=0)
        df = df.rename_axis('cidx').reset_index()
        df = df.assign(num=lambda x: x['gamma_hat']*x['den'])
        # Calculate the effect proportion for each
        df = df.assign(ratio=lambda x: x['den'] / n)
        # Get CI
        df = get_CI(df, 'num', 'den', method='beta', alpha=self.alpha)
        df.drop(columns=['num'], inplace=True)
        df.rename(columns={'lb':'gamma_lb','ub':'gamma_ub'}, inplace=True)
        return df


    def statistic_pval(self, y, s, gamma0):
        """
        Get the statistic p-value for a given null hypothesis (gamma0)
        """
        df_stat = self.statistic_CI(y, s, self.threshold)
        cn_keep = ['cidx', 'm', 'gamma_hat', 'den']
        df_stat = df_stat[cn_keep]
        sig0 = np.sqrt( (gamma0 * (1-gamma0)) / df_stat['den'] )
        z = (df_stat['gamma_hat'] - gamma0) / sig0
        df_stat = df_stat.assign(z=z, pval = norm.cdf(-z))
        df_stat = df_stat.assign(reject=lambda x: x['pval'] < self.alpha)
        return df_stat
        

    def set_threshold(self, y, s, gamma1, gamma2=None):
        """
        Find the operationg threshold which optains an empirical target of gamma1 of test scores/labels

        Parameters
        ----------
        y:              Binary labels
        s:              Scores
        gamma1:         Performance measure target for m1
        gamma2:         If not None, overrides gamma1 and finds operating threshold for m2 instead

        Returns
        -------
        self.threshold:     threshold matches dimension of s
        self.df:            (1-alpha)% CI on underlying performance measure
        """
        g1, g2 = gamma1, gamma1
        if gamma2 is not None:
            g1, g2 = gamma2, gamma2
        # Set performance measure
        self.m1 = self.m1(gamma=g1, alpha = self.alpha)
        self.m2 = self.m2(gamma=g2, alpha = self.alpha)
        # Find operating threshold for gamma{i}
        if gamma2 is not None:
            self.threshold = self.m2.learn_threshold(y, s, method='point')
        else:
            self.threshold = self.m1.learn_threshold(y, s, method='point')
        # Get statistic and CI
        self.df = self.statistic_CI(y, s, self.threshold)
        self.df.drop(columns=['den'], inplace=True)
        # Add on threshold
        if isinstance(self.threshold, float):
            tt = np.array([self.threshold])
        else:
            tt = self.threshold.values
        self.df = self.df.assign(threshold=np.tile(tt, [2,1]).flat)


    def get_power(self, n_trial, margin, adjust=True):
        """
        Calculates the 95% CI for power using uncertainty around gamma
        
        Parameters
        ----------
        n_trial:            Number of trial sample observations
        margin:             Null hypothesis margin between gamma_hat and gamma0
        adjust:             Whether n_trial should be adjusted by the denominator ratio
        """
        n_trial_eff = n_trial
        if adjust:  # Adjust n_trial by the denominator ratio
            n_trial_eff = n_trial * self.df['ratio']
        # Set up null hypothesis
        gamma_hat = self.df['gamma_hat'].values
        gamma_ub = self.df['gamma_ub']
        gamma_lb = self.df['gamma_lb']
        gamma0 = gamma_hat - margin
        assert np.all(gamma0 >= 0), 'margin is too high! one null is less than zero'
        # Calculate power range
        power_point = power_binom(spread=margin, n_trial=n_trial_eff, gamma=gamma_hat, gamma0=gamma0, alpha=self.alpha)        
        power_ub = power_binom(spread=margin, n_trial=n_trial_eff, gamma=gamma_ub, gamma0=gamma0, alpha=self.alpha)
        power_lb = power_binom(spread=margin, n_trial=n_trial_eff, gamma=gamma_lb, gamma0=gamma0, alpha=self.alpha)
        # Combine and return
        power_ci = pd.DataFrame({'gamma0':gamma0,'power_point':power_point, 'power_lb':power_lb, 'power_ub':power_ub})
        power_ci = pd.concat(objs=[self.df, power_ci], axis=1)        
        return power_ci

