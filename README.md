# Files in MLStatEval

Most files in MLStatEval are for internal use. The two main classes are to be found in `trial.py`: `classification` and `regression`.

In the main folder (where `.git` lives), run `source MLStatEval/simulations/pipeline.sh`.

A new performance measure class can be added to `MLStatEval.utils.performance`. The class must have the following structure:

1. Method `statistic(y, s, threshold)` which returns an array of the statistic equivalent in length to then number of columns of the scores or labels.
2. Method `learn_threshold(y, s, gamma, n_bs=1000, seed=None)` which returns an array of threshold designed to reach some `gamma` target.