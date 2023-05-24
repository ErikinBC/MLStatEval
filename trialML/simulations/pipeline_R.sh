#!/bin/bash

# This gets the folder where pipeline.sh lives
dir_here=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "pipeline is here = "$dir_here

# (1) Get the number of arXiv papers
Rscript $dir_here/r1_get_arxiv.R $dir_here
#   output:     ~/figures/gg_arxiv.png
#               ~/data/df_arxiv.csv.png

# (2) Get the number of FDA approvals
Rscript $dir_here/r2_get_fda.R $dir_here
#   output:     ~/figures/gg_fda.png

echo "~~~ End of pipeline_R.sh ~~~"