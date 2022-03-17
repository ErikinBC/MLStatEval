# README

The `power` and `trial` modules represent the main modules that should be used by users in their analysis. The other functions are largely for internal uses.

## Adding new classification performance functions

Performance functions are called in the `trial.py` script from `utils.m_classification` and stored in `di_performance`. To add a new performance function, create a class in the `utils.m_classification` script and ensure it has a `statistic` and `learn_threshold` method and an initialization of `alpha` and `gamma`. The `statistic` method should calculate (a vectorized) statistic using `y`, `s`, and `threshold`.  The `learn_threshold` method should either return the point estimate to target `gamma` or one of three bootstrapping approaches: basic, percentile, and bca. 
