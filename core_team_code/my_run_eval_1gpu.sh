#!/bin/bash
# =====================================================================
# Closed-loop evaluation script for online DPMM models (1 GPU, Bench2Drive).
#
# Usage:
#   bash core_team_code/my_run_eval_1gpu.sh
#
# Set TEAM_CONFIG to your ability subdirectory containing
# config.json + latest.pth (or model_XXXX.pth).
# =====================================================================
set -euo pipefail

export PROJECT_ROOT=/home/spc/syb_carla_garage
export CARLA_ROOT=/home/spc/carla_0_9_15
export WORK_DIR=${PROJECT_ROOT}/Bench2Drive

export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}/core_team_code
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}/team_code
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/spc/anaconda3/lib

BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=${WORK_DIR}/leaderboard/data/bench2drive220

# --- KEY SETTINGS: adjust these ---
TEAM_AGENT=${PROJECT_ROOT}/core_team_code/my_online_dpmm_agent.py

# Point to one ability's model directory:
#   e.g. log/syb_online_dpmm_1gpu_testMini/Emergency_Brake
TEAM_CONFIG=${PROJECT_ROOT}/log/syb_online_dpmm_1gpu_testMini/Emergency_Brake

BASE_CHECKPOINT_ENDPOINT=my_eval_b2d220
PLANNER_TYPE=online_dpmm
ALGO=online_dpmm
SAVE_PATH=${WORK_DIR}/leaderboard/data/results/eval_b2d220_${ALGO}
RESULT_PATH=online_dpmm

if [ ! -d "${RESULT_PATH}" ]; then
    mkdir ${RESULT_PATH}
    echo -e "\033[32m Directory ${RESULT_PATH} created. \033[0m"
fi

echo -e "**************\033[36m Running on single GPU \033[0m **************"
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${WORK_DIR}/${RESULT_PATH}/${BASE_CHECKPOINT_ENDPOINT}.json"
mkdir -p "${WORK_DIR}/${RESULT_PATH}"
GPU_RANK=0

echo -e "\033[32m TEAM_AGENT:  $TEAM_AGENT \033[0m"
echo -e "\033[32m TEAM_CONFIG: $TEAM_CONFIG \033[0m"
echo -e "\033[32m ROUTES:      $ROUTES \033[0m"
echo -e "\033[32m RESULT:      $CHECKPOINT_ENDPOINT \033[0m"

bash -e ${WORK_DIR}/leaderboard/scripts/run_evaluation.sh \
    $BASE_PORT $BASE_TM_PORT $IS_BENCH2DRIVE \
    $ROUTES $TEAM_AGENT $TEAM_CONFIG \
    $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK