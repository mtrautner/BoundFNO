#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=64G
#SBATCH -J "train-smooth"    # job name
#SBATCH --output=outputs/train-smooth-xy.out 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=1
cd ../

python  -u train_fno.py 'smooth_x_y_grid'

