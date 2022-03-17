# Toy example of model trained on random data
import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(1234)
n, p, k = 100, 10, 50
X, y = np.random.randn(n, p), np.random.binomial(1, 0.5, n)
X_train, y_train, X_test, y_test = X[:k], y[:k], X[k:], y[k:]
mdl = LogisticRegression(penalty='none', solver='lbfgs')
mdl.fit(X=X_train, y=y_train)



