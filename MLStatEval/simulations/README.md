# simulations folder

The scripts in this folder carry out the various simulations needed to generate the figures found in the [StatMLEval]() paper.

1. `pipeline_R.sh`: Scrapes web data to show publications/approvals
	I. `r1_get_arxiv.R`: gets arXiv publications with cateogires stat.ML, cs.AI, & cs.LG and produces `figures/gg_arxiv.png` and `data/df_arxiv.csv`.
	II. `r2_get_fda.R`: Gets the number of FDA approvals for SAMD with ML/AI and produces `figures/gg_fda.png` and `data/df_fda.csv`.
2. `pipeline_py.sh` creates the conda environment (`set_env.sh`) and then generates various figures.
	I. `p1_gen_roc.py`: Generates the ROC figures `figures/{gg_roc_gt,gg_auc_gt}.png`.
	II. 
	III. 

## Adding new classification performance functions

Performance functions are called in the `trial.py` script from `utils.m_classification` and stored in `di_performance`. To add a new performance function, create a class in the `utils.m_classification` script and ensure it has a `statistic` and `learn_threshold` method and an initialization of `alpha` and `gamma`. The `statistic` method should calculate (a vectorized) statistic using `y`, `s`, and `threshold`.  The `learn_threshold` method should either return the point estimate to target `gamma` or one of three bootstrapping approaches: basic, percentile, and bca. 