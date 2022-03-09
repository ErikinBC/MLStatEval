#!/bin/bash

# This gets the folder where pipeline.sh lives
dir_here=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "pipeline is here = "$dir_here

# Load the conda environment
env_name=MLStatEval
source $dir_here/set_env.sh $dir_here $env_name

# # ---- (1) Rscripts ---- #
# # (i) Get the number of arXiv papers
# Rscript $dir_here/get_arxiv.R $dir_here
# #   output:     ~/figures/gg_arxiv.png

# # (ii) Get the number of FDA approvals
# Rscript $dir_here/get_fda.R $dir_here
# #   output:     ~/figures/gg_fda.png
# # ---------------------- #


# ---- (2) Python ---- #

python -m MLStatEval.run

# -------------------- #

echo "~~~ End of pipeline.sh ~~~"