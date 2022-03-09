"""
Classes for carrying out inference for classificaiton or regression
"""

import numpy as np

# Internal methods
from MLStatEval.utils.utils import check01, check_binary
from MLStatEval.utils.performance import sensitivity, specificity, precision

di_performance = {'sensitivity':sensitivity, 'specificity':specificity, 'precision':precision}

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
        di_keys = list(di_performance)
        assert m in di_keys, 'performance measure (m) must be one of: %s' % di_keys
        self.m = di_performance[m]
        attrs = ['learn_threshold', 'statistic']
        for attr in attrs:
            hasattr(self.m, attr), 'performance measure (m) needs to have attribute %s' % attr
        # Valid methods for other methods
        self.lst_threshold_method = ['empirical', 'bootstrap', 'umbrella']
        self.lst_power_method = ['one-sided', 'two-sided']
        # self.lst_bs_method = ['']


    def set_threshold(self, y, s, method, **args):
        """
        Learn threshold to optimize performance measure
        
        Inputs:
        y:                      Binary labels
        s:                      Scores
        m:                      Performance measure

        Outputs:
        self.threshold_hat:     Data-derived threshold with method for each column (k)
        """
        assert method in self.lst_threshold_method, 'threshold method must be one of: %s' % self.lst_threshold_method
        self.threshold_method = method
        assert check_binary(y), 'y must be array/matrix of only {0,1}!'
        assert len(y) == len(s), 'y and s must be the same length!'
        self.threshold_hat = self.m.learn_threshold(y=y, s=s, method=method, gamma=self.gamma, **args)
        

    def calculate_power(self, method='one-sided'):
        assert hasattr(self, 'threshold_method'), 'set_threshold method needs to be called before calculated_power'
        assert method in self.lst_power_method, 'power method must be one of %s' % self.lst_power_method




# class regression():
#     """
#     Main class for supporting statistical calibration of ML models for regression task
#     """
#     def __init__(self):
#         None
