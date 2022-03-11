#!/bin/bash

# This gets the folder where pipeline.sh lives
dir_here=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "pipeline is here = "$dir_here

echo "Load the conda environment"
env_name=MLStatEval
source $dir_here/set_env.sh $dir_here $env_name

echo "(1) Generate ROC figures"
python -m MLStatEval.simulations.p1_gen_roc

echo "(2) Evaluating threshold & power inferences"
python -m MLStatEval.simulations.p2_threshold_power

# echo "(3) "
# python -m MLStatEval.simulations.p3_


echo "~~~ End of pipeline.sh ~~~"