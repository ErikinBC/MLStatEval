"""
Classes for carrying out inference for classificaiton or regression
"""

import numpy as np
from MLStatEval.utils.utils import check01, check_binary

class classification():
    """
    Main class for supporting statistical calibration of ML models for classification task
    
    gamma:      Target performance measure
    alpha:      Type-I error rate
    m:          Performance measure
    """
    def __init__(self, gamma, alpha, m):
        assert check01(gamma), 'gamma needs to be between (0,1)!'
        assert check01(alpha), 'alpha needs to be between (0,1)!'
        self.gamma = gamma
        self.alpha = alpha
        # Call in the performance function
        self.m = m
        # Valid methods for other methods
        self.lst_threshold_method = ['empirical', 'bootstrap', 'umbrella']
        self.lst_bs_method = ['']
        self.lst_power_method = ['one-sided', 'two-sided']

    def normal_theory(self, mu1=None, mu0=None, sd1=None, sd0=None, empirical=False):
        if empirical:
            s1 = np.array(self.s[self.y == 1])
            s0 = np.array(self.s[self.y == 0])
            self.mu1, self.sd1 = s1.mean(), s1.std(ddof=1)
            self.mu0, self.sd0 = s0.mean(), s0.std(ddof=0)
        else:
            self.mu1, self.sd1 = mu1, sd1
            self.mu0, self.sd0 = mu0, sd0

    def gen_theory(self):
        assert hasattr(self, 's1'), 'Run normal_theory before gen_theory'

    def set_threshold(self, y, s, method):
        """
        Learn threshold to optimize performance measure
        
        y:          Binary labels
        s:          Scores
        m:          Performance measure
        """
        assert method in self.lst_threshold_method, 'threshold method must be one of: %s' % self.lst_threshold_method
        self.threshold_method = method
        assert check_binary(y), 'y must be array/matrix of only {0,1}!'
        assert len(y) == len(s), 'y and s must be the same length!'
        # self.y, self.s = y, s

    def calculate_power(self, method='one-sided'):
        assert hasattr(self, 'threshold_method'), 'set_threshold method needs to be called before calculated_power'
        assert method in self.lst_power_method, 'power method must be one of %s' % self.lst_power_method


class regression():
    """
    Main class for supporting statistical calibration of ML models for regression task
    """
    def __init__(self):
        None
