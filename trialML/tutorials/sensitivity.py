# Load modules
import numpy as np
from sklearn.linear_model import LogisticRegression
from trialML.trial import classification

## (1) Train a model and obtain scores on a test set
np.random.seed(1)
n, p = 150, 10
k1, k2 = 50, 100
X, y = np.random.randn(n, p), np.random.binomial(1, 0.5, n)
X_train, y_train, X_test, y_test = X[:k1], y[:k1], X[k1:k2], y[k1:k2]
mdl = LogisticRegression(penalty='none', solver='lbfgs')
mdl.fit(X=X_train, y=y_train)
# test set scores
s_test = mdl.predict_proba(X_test)[:,1]
s_test = np.log(s_test / (1-s_test))  # logit transform

## (2) Calibrate operating threshold to achieve 50% sensitivity, 95% of the time
gamma = 0.5  # performance measure target
alpha = 0.05  # type-I error rate for threshold selection
m = 'sensitivity'  # currently supports sensitivity/specificity/precision

# Set up statistical tool
calibration = classification(gamma=gamma, alpha=alpha, m=m)
# Learn threshold
calibration.learn_threshold(y=y_test, s=s_test, method='percentile', n_bs=1000, seed=1)
# Observe test-set performance
gamma_hat_test = calibration.statistic(y=y_test, s=s_test, threshold=calibration.threshold_hat)
print('Empirical sensitivity on test-set: %0.1f%%' % (100*gamma_hat_test))

## (3) Estimate power for trial data
X_trial, y_trial = X[k1:], y[k1:]
n_trial = len(X_trial)
gamma0 = 0.45
spread = gamma - gamma0

calibration.calculate_power(spread, n_trial, threshold=calibration.threshold_hat)
print('Expected trial power for a %0.1f%% margin is at least %0.1f%%' % (100*spread, 100*calibration.power_hat))

## (4) Run trial
s_trial = mdl.predict_proba(X_trial)[:,1]
s_trial = np.log(s_trial / (1-s_trial))  # logit transform
gamma_trial, pval_trial = calibration.statistic(y=y_trial, s=s_trial, gamma0=gamma0, threshold=calibration.threshold_hat)
print('Trial sensitivity: %0.1f%%, trial null-hypothesis: %0.1f%%, trial p-value: %0.5f' % (100*gamma_trial, 100*gamma0, pval_trial))