#!/bin/bash
#SBATCH --array=1-74
#SBATCH --partition=long  # Ask for unkillable job
#SBATCH --cpus-per-task=6                    # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=8Gb                             # Ask for 10 GB of RAM
#SBATCH --time=23:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/nicaandr/slurm_logs/slurm-%j.out  # Write the log on tmp1

# 1. Load your environment
source $HOME/.bashrc
module load anaconda/3 python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0
module load tensorflow
conda activate andreic
source deactivate
source activate andreic
conda activate andreic
echo "PYTHON::::"
which python

# rel_path="/network/tmp1/username/PATH_TO_EXP_FOLDER"
#rel_path="/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/results/2020Jul14-213435_default/"
#rel_path="/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/results/2020Jul14-221038_default/"
#rel_path="/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/results/2020Jul14-222704_default/"
#rel_path="/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/results/2020Jul15-002922_resnets/"
#rel_path="/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/results/2020Jul15-093802_resnets/"
#rel_path="/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/results/2020Jul15-101121_resnets/"
rel_path="/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/results/2020Jul15-102058_resnets/"

echo "Running sbatch array job $SLURM_ARRAY_TASK_ID"

DISABLE_MUJOCO_RENDERING=1 liftoff train_main.py --max-runs 1 --no-detach ${rel_path}
