#!/bin/bash
#SBATCH --job-name=copy # Number of tasks (see below)
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=18         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --output=%j.out  # File to which STDOUT will be written
#SBATCH --error=%j.err   # File to which STDERR will be written
#SBATCH --partition=2080-galvani

# print info about current job

scontrol show job $SLURM_JOB_ID


rclone copy --sftp-host 134.2.168.205 --sftp-user bjaeger25 --sftp-key-file /mnt/qb/home/geiger/bjaeger25/.ssh/VM_geiger_A100_0 --sftp-known-hosts-file /mnt/qb/home/geiger/bjaeger25/.ssh/known_hosts  /mnt/qb/work/geiger/bjaeger25/data/garage_2_cleanup/results/data/garage_v0_2024_10_24 :sftp:/weka/geiger/bjaeger25/garage_2_cleanup/results/data/garage_v0_2024_10_24 --progress --multi-thread-streams=18