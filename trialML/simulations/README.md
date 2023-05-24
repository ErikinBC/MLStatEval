# simulations folder

The scripts in this folder carry out the various simulations needed to generate the figures found in the trialML (UNDER CONSTRUCTION) paper.

1. `pipeline_R.sh`: Scrapes web data to show publications/approvals
	I. `r1_get_arxiv.R`: gets arXiv publications with cateogires stat.ML, cs.AI, & cs.LG and produces `figures/gg_arxiv.png` and `data/df_arxiv.csv`.
	II. `r2_get_fda.R`: Gets the number of FDA approvals for SAMD with ML/AI and produces `figures/gg_fda.png` and `data/df_fda.csv`.
2. `pipeline_py.sh` creates the conda environment (`set_env.sh`) and then generates various figures.
	I. `p1_gen_roc.py`: Generates the ROC figures `figures/{gg_roc_gt,gg_auc_gt}.png`.
	II. `p2_thresold_power.py`: Generates plots to show coverage of threshold choice and powe estimates `figures/{gg_power_comp,gg_threshold_method}.png`.
	III. `p3_precision.py`: Generates (non) monotonic precision/threshold trade-offs `figures/gg_ppv.png`.
	IV. `p4_power_twosided.py`: Generates figures should the 95% confidence intervals for the power estimates `figures/{gg_power_coverage, gg_power_margin}.png`.

