#!/bin/bash

# Assign variables
dir_here=$1
env_name=$2

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


# Check an activate R path exists
path_R=$(which R)
n_char=$(echo $path_R | wc -w)
if [[ "$n_char" -eq 0 ]]; then
    echo "Error! An installed of R was not found"
    return
fi


echo "~~~ End of set_env.sh ~~~"
