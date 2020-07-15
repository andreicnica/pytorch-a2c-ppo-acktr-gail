#!/bin/bash
#SBATCH --array=1-NUM_EXPS
#SBATCH --partition=long  # Ask for unkillable job
#SBATCH --cpus-per-task=6                    # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=8Gb                             # Ask for 10 GB of RAM
#SBATCH --time=23:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/nicaandr/slurm_logs/slurm-%j.out  # Write the log on tmp1

# 1. Load your environment
source /etc/profile
module load tensorflow
module load anaconda/3 python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0
#source $CONDA_ACTIVATE
conda-activate
source activate andreic
conda deactivate
source activate andreic


# rel_path="/network/tmp1/username/PATH_TO_EXP_FOLDER"

echo "Running sbatch array job $SLURM_ARRAY_TASK_ID"

DISABLE_MUJOCO_RENDERING=1 liftoff train_main.py --max-runs 1 --no-detach ${rel_path}