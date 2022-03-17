# trialML: Preparing a machine learning model for a statistical trial

`trialML` is a `python` package designed to help researchers and practioners prepare their machine learning models for a statistical trial to establish a lower-bound on model performance. Specifically, this package helps to calibrating the operating threshold of a binary classifier and carry out a power analysis. 

## Features

The main modules from `trialML` can be called in with one line of code: `from trialML import trial, power`. Their key methods are outlined below, and are described with more detail by the docstrings (e.g. `help('trialML.trial.classification')`). 

1. `trial.classification(gamma, m, alpha)`: determine optimal threshold and calculate power of future trial
    I. `statistic(y, s, threshold, pval=False)`: return the performance measure for a given threshold (and possibly p-value from null hypothesis)
    II. `learn_threshold(y, s, method='percentile', n_bs=1000, seed=None, inherit=True)`
    III. `calculate_power`
2. `power.twosided_classification`: estimate power range (confidence interval) for future trial
    I. `set_threshold`
    II. `statistic_CI`
    III. `statistic_pval`

## How to use

The code block below shows how to calibrate a classifier for toy example of a classifier trained on random data. For more detailed examples wiht real data, please see the [tutorials](https://github.com/ErikinBC/trialML/tree/main/trialML/tutorials) folder.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(1234)
n, p, k = 100, 10, 50
X, y = np.random.randn(n, p), np.random.binomial(1, 0.5, n)
X_train, y_train, X_test, y_test = X[:k], y[:k], X[k:], y[k:]
mdl = LogisticRegression(penalty='none', solver='lbfgs')
mdl.fit(X=X_train, y=y_train)
```

## How to install

`trialML` is available on [PyPI]() can be installed in one line: `pip install trialML`.


