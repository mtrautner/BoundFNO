#!/bin/bash

#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:0
#SBATCH --mem-per-cpu=64G
#SBATCH -J "gen-GRF"    # job name
#SBATCH --output=outputs/gen-GRF.out 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=0-4

cd ../

let N=2

let s=$SLURM_ARRAY_TASK_ID+1

python  -u gen_GRF.py $s $N 


# python -u gen_GRF.py $SLURM_ARRAY_TASK_ID 50 128

# PARRAY=(10 50 250 1000 2000 4000 6000 8000 9500)

# for ip1 in {0..8} # 9 options
# do 
#   for i in {0..4} # 5 samples
#   do 
#      let task_id=$ip1*5+$i
#      printf $task_id"\n"
#      if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
#      then
#         data_size=${PARRAY[$ip1]}
# 	    python  -u train_FNM_v2f_model.py ./configs/data_size_configs_v2f/vor_v2f_data_size_${data_size}.yaml $i
#      fi
#   done
# done