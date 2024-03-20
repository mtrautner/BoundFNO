#!/bin/bash

#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=64G
#SBATCH -J "eval-model"    # job name
#SBATCH --output=outputs/eval-model.out 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=1-3
cd ../analysis

python  -u eval_model_trained.py $SLURM_ARRAY_TASK_ID

