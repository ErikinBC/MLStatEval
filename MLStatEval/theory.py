# Class to support generation of Gaussian mixture
from this import d
import numpy as np

# Internal methods
from MLStatEval.utils.utils import check_binary, check01
from MLStatEval.utils.performance import sensitivity, specificity, precision

di_performance = {'sensitivity':sensitivity, 'specificity':specificity, 'precision':precision}

class gaussian_mixture():
    def __init__(self, m=None):
        """
        m:              Performance measure
        """
        di_keys = list(di_performance)
        assert m in di_keys, 'performance measure (m) must be one of: %s' % di_keys


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


    def set_ys(self, y, s):
        """
        Assign labels and scores to class

        y:              Labels {0,1}
        s:              Scores        
        """
        assert check_binary(y), 'y must be array/matrix of only {0,1}!'
        assert len(y) == len(s), 'y and s must be the same length!'
        self.y, self.s = y, s

    
    def oracle_to_threshold(self):
        1

    def threshold_to_oracle(self):
        1

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
        
