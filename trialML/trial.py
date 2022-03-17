"""
Classes for carrying out inference for classificaiton or regression
"""

# Libraries
import numpy as np
import pandas as pd
from scipy.stats import norm

# Internal methods
from trialML.utils.utils import check01, check_binary, get_cn_idx
from trialML.utils.m_classification import sensitivity, specificity, precision
from trialML.utils.m_classification import lst_method as learn_threshold_methods

# Store all classification performance functions in a dictionary
di_m_classification = {'sensitivity':sensitivity, 'specificity':specificity, 'precision':precision}

# Define mandatory methods each classification method is required to have
classification_methods = ['learn_threshold', 'statistic']

"""
Methods for calculating power

one-sided:              Assume gamma is true (nothing stochastic)
two-sided:              Use distribution of of thresholds to estimate power
"""
power_method = ['one-sided', 'two-sided']

class classification():
    """
    Main class for supporting statistical calibration of ML models for classification task
    
    gamma:      Target performance measure
    alpha:      Type-I error rate
    m:          Performance measure
    m2:         Second performance measure (for stochastic power estimate)
    """
    def __init__(self, gamma, m, alpha=0.05):
        assert check01(gamma), 'gamma needs to be between (0,1)!'
        assert check01(alpha), 'alpha needs to be between (0,1)!'
        self.gamma = gamma
        self.alpha = alpha
        # Call in the performance function
        di_keys = list(di_m_classification)
        assert m in di_keys, 'performance measure (m) must be one of: %s' % di_keys
        self.m = di_m_classification[m](gamma=gamma, alpha=alpha)
        for method in classification_methods:
            hasattr(self.m, method), 'performance measure (m) needs to have method %s' % method


    def statistic(self, y, s, threshold, gamma0=None):
        """
        y:                      Binary labels
        s:                      Scores
        threshold:              Operating threshold (if 2d array, threshold.shape[0] == s.shape[1]) and columns correspond to some different some of method
        gamma0:                 Null hypothesis (if provided returns a p-value)
        """
        m_hat, den_hat = self.m.statistic(y=y, s=s, threshold=threshold, return_den=True)
        if gamma0 is not None:
            cn, idx = get_cn_idx(m_hat)
            sig0 = np.sqrt( (gamma0 * (1-gamma0)) / den_hat )
            z = (m_hat - gamma0) / sig0
            pval = norm.cdf(-z)
            if isinstance(cn, list):
                pval = pd.DataFrame(pval, columns = cn, index=idx)
            return m_hat, pval
        else:
            return m_hat


    def learn_threshold(self, y, s, method='percentile', n_bs=1000, seed=None, inherit=True):
        """
        Learn threshold to optimize performance measure
        
        Inputs:
        y:                      Binary labels
        s:                      Scores
        method:                 A valid inference method (see lst_method)
        n_bs:                   # of bootstrap iterations
        seed:                   Seeding results
        inherit:                Whether y/s should be stored in class

        Outputs:
        self.threshold_hat:     Data-derived threshold with method for each column (k)
        """
        if isinstance(method, str):
            method = [method]
        assert all([meth in learn_threshold_methods for meth in method]), 'Ensure that method is one of: %s' % learn_threshold_methods
        self.threshold_method = method
        assert check_binary(y), 'y must be array/matrix of only {0,1}!'
        assert len(y) == len(s), 'y and s must be the same length!'
        self.threshold_hat = self.m.learn_threshold(y=y, s=s, method=method, n_bs=n_bs, seed=seed)
        if inherit:
            self.y, self.s = y, s        

    def calculate_power(self, spread, n_trial, threshold=None):
        """
        Calculate expected power for a future trial. If y/s/threshold are supplied, then n_trial will be weighted by yhat to adjust for the sample size
        """
        attrs = ['threshold_method', 'y', 's']
        for attr in attrs:
            assert hasattr(self, attr), 'attribute %s not found\nlearn_threshold with inherit=True needs to be called before calculated_power' % attr
        # If not threshold is given, n_trial will assume to be class-specific
        n_trial_exp = n_trial
        if threshold is not None:
            n = len(self.y)
            _, den_hat = self.m.statistic(y=self.y, s=self.s, threshold=threshold, return_den=True)
            n_trial_exp = n_trial * den_hat / n
        self.power_hat = self.m.estimate_power(spread=spread, n_trial=n_trial_exp)
        

