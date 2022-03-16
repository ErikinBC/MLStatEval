# Class to support generation of Gaussian mixture
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

# Internal methods
from trialML.utils.m_classification import sensitivity, specificity, precision
from trialML.utils.utils import check_binary, check01, get_cn_idx, array_to_float, to_array
from trialML.utils.theory import oracle_auroc, threshold_to_sensitivity, sensitivity_to_threshold, specificity_to_threshold, threshold_to_specificity, threshold_to_precision, precision_to_threshold

di_performance = {'sensitivity':sensitivity, 'specificity':specificity, 'precision':precision}

# self = gaussian_mixture()
# self.set_params(p=0.5,mu1=2,mu0=1,sd1=1,sd0=1,empirical=False)
class gaussian_mixture():
    def __init__(self) -> None:
        pass

    def set_params(self, p=None, mu1=None, mu0=None, sd1=None, sd0=None, empirical=False):
        """
        Class to support generation of Gaussian mixtures.

        p:              P(y==1)
        mu{j}:          Mean of class {j}
        sd{j}:          Standard deviation of class {j}
        empirical:      Should terms above estimated from inherited y/s?
        """
        if empirical:
            assert hasattr(self, 's'), 'set_threshold needs to be run if you want to estimate values empirically'
            self.p = np.mean(self.y)
            s1 = np.array(self.s[self.y == 1])
            s0 = np.array(self.s[self.y == 0])
            self.mu1, self.sd1 = s1.mean(), s1.std(ddof=1)
            self.mu0, self.sd0 = s0.mean(), s0.std(ddof=0)
        else:
            assert check01(p), 'p needs to be between (0,1)'
            assert mu1 > mu0, 'Mean of class 1 needs to be greater than class 0!'
            assert (sd1 > 0) & (sd0 > 0), 'Std. dev needs to be greater than zero!'
            self.p = p
            self.mu1, self.sd1 = mu1, sd1
            self.mu0, self.sd0 = mu0, sd0
        # Oracle AUROC
        self.auroc = oracle_auroc(self.mu1, self.mu0, self.sd1, self.sd0)


    def set_ys(self, y, s):
        """
        Assign labels and scores to class

        y:              Labels {0,1}
        s:              Scores        
        """
        assert check_binary(y), 'y must be array/matrix of only {0,1}!'
        assert len(y) == len(s), 'y and s must be the same length!'
        self.y, self.s = y, s

    def gen_roc_curve(self, n_points=500, ptail=1e-3):
        """
        Generate sequence a sensitivity/specificity trade-off points for a curve

        n_points:           Number of points to evaluate
        ptail:              What percentile in the tail to start/stop the sequence
        """
        s_lower = norm.ppf(ptail)
        s_upper = norm.ppf(1-ptail) + self.mu
        s_seq = np.linspace(s_lower, s_upper, n_points)
        sens = threshold_to_sensitivity(s_seq)
        spec = threshold_to_specificity(s_seq)
        res = pd.DataFrame({'thresh':s_seq, 'sens':sens, 'spec':spec})
        return res


    def gen_pr_curve(self, n_points=100, ptail=0.001):
        """
        Generate a sequence of precison/recall trade-off points for a curve

        n_points:           Number of points to evaluate
        ptail:              What percentile in the tail to start/stop the sequence
        """
        z_alpha = norm.ppf(1-ptail)
        # (i) Plotting ranges
        plusminus = np.array([-1, 1])*z_alpha
        b0 = self.mu1 + plusminus*self.sd0
        b1 = self.mu0 + plusminus*self.sd1
        lb = min(min(b0),min(b1))
        ub = max(max(b0),max(b1))
        thresh_seq = np.linspace(lb, ub, n_points)
        ppv = threshold_to_precision(thresh_seq, self.mu1, self.mu0, self.sd1, self.sd0, self.p)
        recall = threshold_to_sensitivity(thresh_seq, self.mu1, self.sd1)
        res = pd.DataFrame({'thresh':thresh_seq, 'ppv':ppv, 'recall':recall})
        return res


    def set_threshold(self, threshold):
        """
        Convert the threshold into the oracle performance values

        Input:
        threshold:          An array/DataFrame of threshold values

        Output:
        self.oracle_m:      Dictionary with different performance metric values
        """
        cn, idx = get_cn_idx(threshold)
        self.threshold = threshold
        # Calculate the oracle performance measures
        oracle_sens = threshold_to_sensitivity(threshold, self.mu1, self.sd1)
        oracle_spec = threshold_to_specificity(threshold, self.mu0, self.sd0)
        oracle_prec = threshold_to_precision(threshold, self.mu1, self.mu0, self.sd1, self.sd0, self.p)
        self.oracle_m = {'sensitivity':oracle_sens, 'specificity':oracle_spec, 'precision':oracle_prec}
        if isinstance(cn, list):
            self.oracle_m = {k:pd.DataFrame(v,columns=cn,index=idx) for k,v in self.oracle_m.items()}
    

    def set_gamma(self, gamma):  #, alpha
        """
        Find the oracle thresholds by setting gamma

        gamma:          Performance target
        """
        assert check01(gamma), 'gamma needs to be between (0,1)'
        oracle_sens = sensitivity_to_threshold(gamma, self.mu1, self.sd1)
        oracle_spec = specificity_to_threshold(gamma, self.mu0, self.sd0)
        oracle_prec = precision_to_threshold(gamma, self.mu1, self.mu0, self.sd1, self.sd0, self.p)
        di_threshold = {'sensitivity':oracle_sens, 'specificity':oracle_spec, 'precision':oracle_prec}
        # Convert to float if relevant
        self.oracle_threshold = {k:array_to_float(v) for k,v in di_threshold.items()}


    def check_threshold_coverage(self, threshold, m):
        """
        Determine whether a given threshold is sufficiently conservative to meet a certain gamma level
        """
        assert hasattr(self, 'oracle_threshold'), 'run set_gamma() before calling check_threshold'
        assert m in self.oracle_threshold, 'm needs to be one of: %s' % list(self.oracle_threshold)
        oracle_threshold = self.oracle_threshold[m]
        if m == 'sensitivity':
            check = threshold <= oracle_threshold
        elif (m == 'specificity') | (m == 'precision'):
            check = threshold >= oracle_threshold
        else:
            sys.exit('How did we get here?!')
        return check

    def gen_mixture(self, n, k=1, seed=None, keep=False):
        """
        Generate scores from a Gaussian mixture N(m1,s1) N(m0,s0)
        
        n:              Number of samples
        k:              Number of simulations (stored as columns)
        seed:           Seed for random draw
        keep:           Should scores and labels be stored as attributes?

        Returns:
        y, s
        """
        assert (n > 0) and isinstance(n, int), 'n needs to be an int > 0'
        if seed is not None:
            np.random.seed(seed)
        y = np.random.binomial(n=1, p=self.p, size=[n, k])
        s = np.random.randn(n, k)
        idx1, idx0 = y == 1, y == 0
        s[idx1] *= self.sd1
        s[idx1] += self.mu1
        s[idx0] *= self.sd0
        s[idx0] += self.mu0
        if keep:
            self.y, self.s = y, s
        else:
            return y, s
        
