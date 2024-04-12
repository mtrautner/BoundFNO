#!/bin/bash

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH -J "train-cell-problem"    # job name
#SBATCH --output=outputs/train-cell-problem-rescale.out 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=slanth@caltech.edu
####SBATCH --qos=debug

cd ../

python -u train_cell_problem.py -c config_default.yaml

