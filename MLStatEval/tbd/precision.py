import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar

"""
Class to solve for aspects of precision
p:          Label proportion
mu1:        Mean of positive class
s1:         Standard error of positive class
"""
class precision():
    def __init__(self, p, mu1, mu0=0, s1=1, s0=1):
        assert mu1 > mu0, 'mu1 > mu0'
        assert (p > 0) & (p < 1)
        self.p = p
        self.mu1, self.mu0 = mu1, mu0
        self.s1, self.s0 = s1, s0

    # For a given threshold, find oracle PPV
    def prec_gauss(self, thresh):
        z1 = (thresh - self.mu1)/self.s1
        z0 = (thresh - self.mu0)/self.s0
        term1 = norm.cdf(-z1)*self.p
        term2 = norm.cdf(-z0)*(1-self.p)
        ppv = term1/(term1+term2)
        return ppv

    # Function to calculate error from PPV target for a given threshold
    def err_prec(self, thresh, target):
        err = target - self.prec_gauss(thresh)
        return err

    # Wrapper to find thresholds for a given target
    def find_ppv_thresh(self, target):
        s_max = max(self.s1, self.s0)
        bounds = 3 * s_max
        thresh = root_scalar(f=self.err_prec, args=(target), x0=0, x1=0.1, method='secant', bracket=(-bounds,+bounds)).root
        return thresh

    # Generate precision/recall for a series of operating thresholds
    def pr_curve(self, n_points=100, alpha=0.001):
        z_alpha = norm.ppf(1-alpha)
        # (i) Plotting ranges
        plusminus = np.array([-1, 1])*z_alpha
        b0 = self.mu1 + plusminus*self.s0
        b1 = self.mu0 + plusminus*self.s1
        lb = min(min(b0),min(b1))
        ub = max(max(b0),max(b1))
        thresh_seq = np.linspace(lb, ub, n_points)
        ppv = self.prec_gauss(thresh_seq)
        recall = norm.cdf((self.mu1 - thresh_seq)/self.s1)
        res = pd.DataFrame({'thresh':thresh_seq, 'ppv':ppv, 'recall':recall})
        return res

    # Generate mills ratio curve
    def mills_curve(self):
        1

    # Find points where precision is declining
    def find_negative(self):
        if self.s1 == self.s0:
            print('When variances are equal, precision is monotonically increasing')
            return np.nan
        elif self.s1 > self.s0:
            print('PPV decreasing from -infty to thresh')
        else:
            print('PPV increasing from thresh to infty')
    


self = precision(p=0.25,mu1=1,mu0=0,s1=1,s0=1)
thresh = self.find_ppv_thresh(0.5)
self.prec_gauss(thresh)


