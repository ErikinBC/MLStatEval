"""
Classes for carrying out inference for classificaiton or regression
"""

import numpy as np

# Internal methods
from MLStatEval.utils.performance import lst_method
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
    def __init__(self, gamma, m, alpha=0.05):
        assert check01(gamma), 'gamma needs to be between (0,1)!'
        assert check01(alpha), 'alpha needs to be between (0,1)!'
        self.gamma = gamma
        self.alpha = alpha
        # Call in the performance function
        di_keys = list(di_performance)
        assert m in di_keys, 'performance measure (m) must be one of: %s' % di_keys
        self.m = di_performance[m](gamma=gamma, alpha=alpha)
        attrs = ['learn_threshold', 'statistic']
        for attr in attrs:
            hasattr(self.m, attr), 'performance measure (m) needs to have attribute %s' % attr
        # Valid methods for other methods
        self.lst_threshold_method = lst_method
        self.lst_power_method = ['one-sided', 'two-sided']

    def statistic(self, y, s, threshold):
        m_hat = self.m.statistic(y=y, s=s, threshold=threshold)
        return m_hat

    def learn_threshold(self, y, s, method='percentile', n_bs=1000, seed=None):
        """
        Learn threshold to optimize performance measure
        
        Inputs:
        y:                      Binary labels
        s:                      Scores
        method:                 A valid inference method (see lst_method)
        n_bs:                   # of bootstrap iterations
        seed:                   Seeding results

        Outputs:
        self.threshold_hat:     Data-derived threshold with method for each column (k)
        """
        self.threshold_method = method
        assert check_binary(y), 'y must be array/matrix of only {0,1}!'
        assert len(y) == len(s), 'y and s must be the same length!'
        self.threshold_hat = self.m.learn_threshold(y=y, s=s, method=method, n_bs=n_bs, seed=seed)
        

    def calculate_power(self, method='one-sided'):
        assert hasattr(self, 'threshold_method'), 'set_threshold method needs to be called before calculated_power'
        assert method in self.lst_power_method, 'power method must be one of %s' % self.lst_power_method




# class regression():
#     """
#     Main class for supporting statistical calibration of ML models for regression task
#     """
#     def __init__(self):
#         None
