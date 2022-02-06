#!/bin/bash

env_name=calib
# Check to see if anaconda/miniconda environment exists
path_conda=$(which conda)
path_conda=$(echo $path_conda | awk '{split($0,a,"3/"); print a[1]}')3
grep_env=$(ls $path_conda/envs | grep $env_name)
n_char=$(echo $grep_env | wc -w)

if [[ "$n_char" -eq 0 ]]; then
    echo "Installing environment"
    # conda create --name $env_name python=3.9
    # conda install -c conda-forge numpy pandas plotnine scikit-learn bottleneck 
    conda create --name $env_name --file conda_env.txt python=3.9
else
    echo "Environment already exists"
    conda list --explicit > conda_env.txt
fi

conda activate $env_name

# Check to see if pip-specific packages are installed
pckgs_pip="patchworklib==0.3.1"
for pckg in $pckgs_pip; do
    echo "package: "$pckg
    grep_env=$(conda env export | grep $pckg)
    n_char=$(echo $grep_env | wc -w)
    if [[ "$n_char" -eq 0 ]]; then
        echo "Installing $pckg"
        pip3 install $pckg
    else
        echo "$pckg already exists"
    fi
done

