#!/bin/bash
#SBATCH --job-name=copy # Number of tasks (see below)
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=32         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --mem=256G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=%j.out  # File to which STDOUT will be written
#SBATCH --error=%j.err   # File to which STDERR will be written
#SBATCH --partition=cpu-galvani

# print info about current job

scontrol show job $SLURM_JOB_ID


rclone copy --sftp-host 134.2.168.216 --sftp-user ubuntu --sftp-key-file /mnt/qb/home/geiger/bjaeger25/.ssh/VM_geiger_A100_0 --sftp-known-hosts-file /mnt/qb/home/geiger/bjaeger25/.ssh/known_hosts  /mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/data/garage_v0_2024_10_22 :sftp:/mnt/bernhard/data/garage_v0_2024_10_22 --progress --multi-thread-streams=32