#!/bin/bash
#SBATCH --job-name=syb_tfpp
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:l40:4  # num_gpu
#SBATCH --mail-type=end
#SBATCH --mail-user=2431997@tongji.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# IMPORTANT: Start this script from within team_code folder, otherwise it will not work
## --load_file /share/home/u19666033/djy/carla_garage/results/BEVDrive_0/model_0030.pth

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd