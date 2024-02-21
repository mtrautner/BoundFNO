#!/bin/bash

#SBATCH --time=0:05:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=64G
#SBATCH -J "eval-model"    # job name
#SBATCH --output=outputs/eval-model.out 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=1-5
cd ../analysis

echo "Running eval_model.py with task id $SLURM_ARRAY_TASK_ID"
python  -u eval_model.py $SLURM_ARRAY_TASK_ID

