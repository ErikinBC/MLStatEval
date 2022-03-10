# simulations folder

The scripts in this folder carry out the various simulations needed to generate the figures found in the [StatMLEval]() paper.

1. `pipeline_R.sh`: scrapes web data to show publications/approvals
	I. `get_arxiv.R`: gets arXiv publications with cateogires stat.ML, cs.AI, & cs.LG and produces `figures/gg_arxiv.png` and `data/df_arxiv.csv`.
	II. `get_fda.R`: gets the number of FDA approvals for SAMD with ML/AI and produces `figures/gg_fda.png` and `data/df_fda.csv`.
2. `pipeline_py.sh` creates the conda environment (`set_env.sh`) and then generates various figures.
	I. `gen_roc.py`:
	II. 
	III. 
