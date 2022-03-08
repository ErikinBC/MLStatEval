#!/bin/bash

# Assign variables
dir_here=$1
env_name=$2

# --- (1) python --- #
# Check to see if anaconda/miniconda environment exists
path_conda=$(which conda)
path_conda=$(echo $path_conda | awk '{split($0,a,"3/"); print a[1]}')3
grep_env=$(ls $path_conda/envs | grep $env_name)
n_char=$(echo $grep_env | wc -w)

if [[ "$n_char" -eq 0 ]]; then
    echo "Installing environment"
    conda env create -f $dir_here/$env_name.yml
else
    echo "Environment already exists"
fi
conda activate $env_name

# --- (2) R --- #

# Set R in conda to path
path_Rscript=$path_conda/envs/MLStatEval/bin/Rscript

echo "~~~ End of set_env.sh ~~~"