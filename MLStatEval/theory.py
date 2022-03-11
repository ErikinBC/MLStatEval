# Class to support generation of Gaussian mixture
from tabnanny import check
import numpy as np
import pandas as pd
from scipy.stats import norm

# Internal methods
from MLStatEval.utils.utils import check_binary, check01, clean_threshold, get_cn_idx
from MLStatEval.utils.m_classification import sensitivity, specificity, precision

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
        # Calculate the ground-truth AUROC
        self.auroc = norm.cdf((self.mu1 - self.mu0) / np.sqrt(self.sd1**2 + self.sd0**2))


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
        Generate sequence of scores within distribution of 0 and 1 class

        n_points:           Number of points to evaluate
        ptail:              What percentile in the tail to start/stop the sequence
        """
        s_lower = norm.ppf(ptail)
        s_upper = norm.ppf(1-ptail) + self.mu
        s_seq = np.linspace(s_lower, s_upper, n_points)
        sens = self.thresh2sens(s_seq)
        spec = self.thresh2spec(s_seq)
        res = pd.DataFrame({'thresh':s_seq, 'sens':sens, 'spec':spec})
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
        oracle_sens = norm.cdf( (self.mu1 - threshold) / self.sd1 )
        oracle_spec = norm.cdf( (threshold - self.mu0) / self.sd0 )
        oracle_prec = self.p*oracle_sens / (self.p*oracle_sens + (1-self.p)*(1-oracle_spec))
        self.oracle_m = {'sensitivity':oracle_sens, 'specificity':oracle_spec, 'precision':oracle_prec}
        if isinstance(cn, list):
            self.oracle_m = {k:pd.DataFrame(v,columns=cn,index=idx) for k,v in self.oracle_m.items()}
    
    def set_gamma(self, gamma, alpha):
        """
        Find the oracle thresholds by setting gamma

        gamma:          Performance target
        """
        assert check01(gamma), 'gamma needs to be between (0,1)'
        assert check01(alpha), 'alpha needs to be between (0,1)'
        thresh_sens = self.mu1 + self.sd1*norm.ppf(1-gamma)
        thresh_spec = self.mu0 + self.sd0*norm.ppf(gamma)
        # thresh_spec = NEED TO IMPLEMENT FUNCTION HERE
        # ID S0 > S1 OR S1 > S0, SEARCH FOR EQUALITY CONDITION OF MILLS RATIO TO FIND TURNING POINT



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
        
