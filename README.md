# trialML: Preparing a machine learning model for a statistical trial

`trialML` is a `python` package designed to help researchers and practioners prepare their machine learning models for a statistical trial to establish a lower-bound on model performance. Specifically, this package helps to:

1. Calibrate the operating threshold of a binary classifier and carry out a power analysis for a specific performance measure (e.g. sensitivity or specificity)
2. Determine the critical value for rejecting the null hypothesis for a regression model's performance (e.g. MSE or MAE) using test set data.

A more formal description of these techniques can be found in the corresponding arXiv paper (UNDER CONSTRUCTION).

<br>

## Features

The main modules from `trialML` can be called in with one line of code: `from trialML import trial, power`. Their key methods are outlined below, and are described with more detail by the docstrings (e.g. `help('trialML.trial.classification')`). 

1. `trial.classification(gamma, m, alpha)`: determine optimal threshold and calculate power of future trial
    1. `statistic(y, s, threshold, pval=..)`: return the performance measure for a given threshold (and possibly p-value from null hypothesis)
    2. `learn_threshold(y, s, method='..')`: calibrate the opearting threshold to obtain at least `gamma` 1-`alpha`% of the time.
    3. `calculate_power(spread, n_trial, threshold)`: estimate power for a given trial sample size and null hypothesis margin (spread). Threshold can be provided to estimate percent of samples that are class-specific.
2. `power.twosided_classification(m1, m2, alpha)`: estimate performance measure and power range (confidence interval) for two performance measures: `m1` and `m2`.
    1. `set_threshold(y, s, gamma1)`: Set the threshold to get a performance level of `gamma1` for the first performance measure `m1`.
    2. `statistic_CI(y, s, threshold)`: Get the (1-`alpha`) confidence interval for the empirical values of `m1` and `m2`.
    3. `statistic_pval(y, s, gamma0)`: Get the p-value on trial data for a given null hypothesis.

<br>

## How to use

The code block below shows how to calibrate a classifier for toy example of a classifier trained on random data. For more detailed examples wiht real data, please see the [tutorials](trialML/tutorials) folder.

```python
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
```
<br>

## How to install

`trialML` is available on [PyPI](https://pypi.org/project/trialML/) can be installed in one line: `pip install trialML`.