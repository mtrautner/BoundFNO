#!/bin/bash

#SBATCH --time=0:30:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --gres gpu:0
#SBATCH --mem-per-cpu=64G
#SBATCH -J "gen-grad-data"    # job name
#SBATCH --output=outputs/gen-grad-data.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=1
cd ../data

python  -u gen_grad_training_data.py


