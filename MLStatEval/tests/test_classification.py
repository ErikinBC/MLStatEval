import unittest
from MLStatEval.trial import classification

class TestClassification(unittest.TestCase):

    def check_init(self):
        # Will be used by later checks
        self.calib = classification(gamma=0.1, alpha=0.05, m=None)
        attrs = ['gamma', 'alpha', 'm']
        for attr in attrs:
            assert hasattr(self.calib, attr), 'classification should have attribute %s' % attr
        

    def check_set_threshold(self):
        y = [0, 1]
        s = [1, 2]
        for method in self.calib.lst_threshold_method:
            self.calib.set_threshold(y, s, method)

    # CHECK THAT get_mixture RETURNS NP ARRAYS 

if __name__ == '__main__':
    unittest.main()