# Load modules
import numpy as np
from sklearn.linear_model import LogisticRegression
from trialML.power import twosided_classification

## (1) Train a model and obtain scores on a test set
np.random.seed(1)
n, p = 150, 10
k1, k2 = 50, 100
X, y = np.random.randn(n, p), np.random.binomial(1, 0.5, n)
X_train, y_train, X_test, y_test = X[:k1], y[:k1], X[k1:k2], y[k1:k2]
X_trial, y_trial = X[k1:], y[k1:]
mdl = LogisticRegression(penalty='none', solver='lbfgs')
mdl.fit(X=X_train, y=y_train)
# test set scores
s_test = mdl.predict_proba(X_test)[:,1]
s_test = np.log(s_test / (1-s_test))  # logit transform

## (2) Select a point on the ROC curve when sensitivity equals 50%
m1 = 'sensitivity'
m2 = 'specificity'
alpha = 0.05  # type-I error rate for test
gamma1 = 0.5  # for sensitivity
power_2s = twosided_classification(m1, m2, alpha)
power_2s.set_threshold(y=y_test, s=s_test, gamma1=gamma1)

## (3) Get performance range
df_gamma = power_2s.statistic_CI(y=y_test, s=s_test, threshold=power_2s.threshold)
df_gamma.round(3)

## (4) Estimate power range
n_trial = len(X_trial)
margin = 0.05
df_power = power_2s.get_power(n_trial=n_trial, margin=margin, adjust=True)
df_power.round(3)

## (5) Run trial
gamma0 = df_gamma['gamma_hat'] - margin
s_trial = mdl.predict_proba(X_trial)[:,1]
s_trial = np.log(s_trial / (1-s_trial))  # logit transform
df_trial = power_2s.statistic_pval(y=y_trial, s=s_trial, gamma0=gamma0)
df_trial.round(3)