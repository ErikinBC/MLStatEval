#!/bin/bash

# Load the conda environment
source set_env.sh

# (1) Get the number of arXiv papers
Rscript get_arxiv.R
#   output:     ~/figures/gg_arxiv.png



# ------ (X) ROC CURVE ------ #

# echo "--- (1) gen_figures.py ---"
# python gen_figures.py

# echo "--- (2) sim_power.py ---"
# # Coverage experiments for power
# python sim_power.py

