#!/bin/bash
#SBATCH --job-name=online_dpmm
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=/share/home/u19666033/syb/carla_garage/log/slurm_logs/online_dpmm_%a_%A.out
#SBATCH --error=/share/home/u19666033/syb/carla_garage/log/slurm_logs/online_dpmm_%a_%A.err
#SBATCH --partition=2080-galvani

# =====================================================================
# SLURM script for online alternating Network-DPMM training (4 GPUs).
#
# Usage:
#   sbatch core_team_code/online_dpmm/my_slurm_train_online_dpmm_4gpu.sh
#
# Adjust the variables below to match your cluster environment.
# =====================================================================

export CARLA_ROOT=/share/home/u19666033/djy/carla2
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/anaconda3/lib
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate syb_garage_2

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1

# --- Paths ---
export PROJECT_ROOT=/share/home/u19666033/syb/carla_garage
LOGDIR=${PROJECT_ROOT}/log
DATASET_ROOT=/share/home/u19666033/syb/pdm_dataset
EXP_ID=syb_online_dpmm_4gpu

# Optional: load a pre-trained model to fine-tune (leave empty to train from scratch)
# LOAD_FILE=${PROJECT_ROOT}/log/syb_tfpp_withB2dTrajFit_stg1/model_0030.pth
LOAD_FILE=""

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

# --- Build the torchrun command ---
CMD="torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d \
  core_team_code/online_dpmm/my_train_ability_wTFFdeQtd_online.py \
  --id ${EXP_ID} \
  --use_disk_cache 0 --crop_image 1 --seed 0 --epochs 31 --batch_size 16 --lr 3e-4 --setting all \
  --root_dir ${DATASET_ROOT} \
  --logdir ${LOGDIR} \
  --use_controller_input_prediction 1 --use_wp_gru 0 --continue_epoch 0 --cpu_cores 32 \
  --freeze_backbone 1 --use_depth 0 --use_semantic 0 --detect_boxes 1 --use_bev_semantic 1 \
  --image_architecture regnety_032 --lidar_architecture regnety_032 \
  --sync_batch_norm 1 \
  --online_dpmm 1 \
  --use_traj_front_door_encoder 1 \
  --use_prior_fuseFeat 1 \
  --traj_dpmm_buffer_size 4096 \
  --fusefeat_dpmm_buffer_size 2048 \
  --dpmm_update_start_step 200 \
  --dpmm_update_freq_steps 0 \
  --traj_dpmm_replay_ratio 0.5 \
  --fusefeat_dpmm_replay_ratio 0.5"

# Optional: add --load_file if specified
if [ -n "${LOAD_FILE}" ]; then
    CMD="${CMD} --load_file ${LOAD_FILE} --continue_epoch 1"
fi

echo "Running command:"
echo "${CMD}"
eval "${CMD}"

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"