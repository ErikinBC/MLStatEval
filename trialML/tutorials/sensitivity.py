"""
Example of how to use model to calibrate model to target 80% sensitivity
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, Binarizer
from trialML.trial import classification

# Load data
X, y = fetch_california_housing(return_X_y=True)
y = np.atleast_2d(y).T

# Split into training, test, and trial
seed = 1234
np.random.seed(seed)
idx_cut = pd.cut(np.random.rand(X.shape[0]),[0,1/3,2/3,1],labels=[0,1,2]).astype(int)
idx_train = (idx_cut == 0)
idx_test = (idx_cut == 1)
idx_trial = (idx_cut == 2)
X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]
X_trial, y_trial = X[idx_trial], y[idx_trial]
n_train, n_test, n_trial = X_train.shape[0], X_test.shape[0], X_trial.shape[0]

# Transform X and y
enc_X = StandardScaler().fit(X_train)
X_train = enc_X.transform(X_train)
X_test = enc_X.transform(X_test)
X_trial = enc_X.transform(X_trial)
med_y = np.median(y_train)
enc_y = Binarizer(threshold=med_y)
enc_y.fit(y_train)
y_train = enc_y.transform(y_train).flatten().astype(int)
y_test = enc_y.transform(y_test).flatten().astype(int)
y_trial = enc_y.transform(y_trial).flatten().astype(int)

# Fit model on training
mdl = LogisticRegression(penalty='none', solver='lbfgs')
mdl.fit(X=X_train, y=y_train)

# Obtain inferences on test
s_test = mdl.predict_proba(X_test)[:,1]

# Learn threshold threshold
gamma = 0.8
m = 'sensitivity'
alpha = 0.05
n_bs = 1000
calibration = classification(gamma=gamma, m=m, alpha=alpha)
calibration.learn_threshold(y=y_test,s=s_test,method='percentile',n_bs=n_bs,seed=seed)
print('Threshold chosen: %0.3f' % calibration.threshold_hat)

# Calculate the empirical sensitivity
m_test = calibration.statistic(y=y_test, s=s_test, threshold=calibration.threshold_hat)
print('Empirical sensitivity on test set %0.1f%% (%0.1f%% target)' % (100*m_test, 100*gamma))

# Estimate power
spread = 0.02  # null hypothesis spread compared to gamma
p_train = y_train.mean()
n_trial_hat = int(p_train * n_trial)  # Estimate the number of positive samples in the trial data
calibration.calculate_power(spread=spread, n_trial=n_trial_hat)
print('Power estimate: %0.1f%%' % (100*calibration.power_hat))

# Run trial
s_trial = mdl.predict_proba(X_trial)[:,1]
m_trial, pval_trial = calibration.statistic(y=y_trial, s=s_trial, threshold=calibration.threshold_hat, pval=True)
reject_null = (pval_trial < alpha)
print('Trial sensitivity: %0.1f%% (%0.1f%% target)\nP-value: %0.4f (reject null=%s)' % (100*m_trial, 100*gamma, pval_trial, reject_null))


print('~~~ End of example.py ~~~')