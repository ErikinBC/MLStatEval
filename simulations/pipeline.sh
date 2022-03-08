#!/bin/bash

# This gets the folder where pipeline.sh lives
dir_here=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "pipeline is here = "$dir_here

# Load the conda environment
env_name=MLStatEval
source $dir_here/set_env.sh $dir_here $env_name
# assigns variable path_Rscript

# ---- (1) Rscripts ---- #
# (i) Get the number of arXiv papers
$path_Rscript $dir_here/get_arxiv.R
#   output:     ~/figures/gg_arxiv.png

# (ii) Get the number of FDA approvals
$path_Rscript $dir_here/get_fda.R
#   output:     ~/figures/gg_fda.png
# ---------------------- #


# # (3) Threshold tools
# python run_threshold.py
# #   output:     ~/

# ------ (X) ROC CURVE ------ #

# echo "--- (1) gen_figures.py ---"
# python gen_figures.py

# echo "--- (2) sim_power.py ---"
# # Coverage experiments for power
# python sim_power.py

echo "~~~ End of pipeline.sh ~~~"