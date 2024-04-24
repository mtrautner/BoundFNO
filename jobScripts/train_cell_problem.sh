#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --gres=gpu:p100:1 # specify gpu type to make timing results comparable
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH -J "train-cell-problem"    # job name
#SBATCH --output=outputs/train-cell-problem.out 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=slanth@caltech.edu
###SBATCH --qos=debug

cd ../

python -u train_cell_problem.py -c config_default.yaml

