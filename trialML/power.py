"""
Classes to support two-sided power calculations
"""

# i) Assign two performance functions / alpha
# ii) Set y, s, and target performance for each
# iii) Power simulations (spread1, spread2, n1, n2)

import pandas as pd

from trialML.utils.stats import get_CI
from trialML.utils.utils import check01, check_binary
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
        self.df:            95%CI on underlying performance measure
        """
        assert check_binary(y), 'y needs to be [0,1]'
        g1, g2 = gamma1, gamma1
        if gamma2 is not None:
            g1, g2 = gamma2, gamma2
        # Set performance measure
        self.m1 = self.m1(gamma=g1, alpha = self.alpha)
        self.m2 = self.m2(gamma=g2, alpha = self.alpha)
        # Find operating threshold for gamma{i}
        if gamma2 is not None:
            threshold = self.m2.learn_threshold(y, s, method='point')
        else:
            threshold = self.m1.learn_threshold(y, s, method='point')
        # Get test statistic for each
        stat1, den1 = self.m1.statistic(y=y, s=s, threshold=threshold, return_den=True)
        stat2, den2 = self.m2.statistic(y=y, s=s, threshold=threshold, return_den=True)
        # Get the lb/ub ranges for the binomial confidence interval
        df1 = pd.DataFrame({'m':1, 'stat':stat1.values.flat, 'den':den1.values.flat})
        df2 = pd.DataFrame({'m':2,'stat':stat2.values.flat, 'den':den2.values.flat})
        df = pd.concat(objs=[df1, df2], axis=0).reset_index(drop=True)
        df = df.assign(num=lambda x: x['stat']*x['den'])
        df = get_CI(df, 'num', 'den', method='beta', alpha=self.alpha)
        # Inherit attributes for later
        self.threshold = threshold
        self.df = df

    def get_power(self, n_trial, margin):
        1
    


# Do some quick tests
import numpy as np
from trialML.theory import gaussian_mixture
normal_dgp = gaussian_mixture()
normal_dgp.set_params(0.5, 1, 0, 1, 1)
y, s = normal_dgp.gen_mixture(50, 25, 1234)
gamma1, gamma2 = 0.5, None
n_trial = 10
margin = np.linspace(0.01,0.2, 20)
self = twosided_classification('sensitivity','specificity',0.05)
self.set_threshold(y=y, s=s, gamma1=0.5, gamma2=None)

self.get_power()
