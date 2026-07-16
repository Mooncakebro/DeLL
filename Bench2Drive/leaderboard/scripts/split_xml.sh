#!/bin/bash
export CARLA_ROOT=/share/home/u19666033/djy/carla2
export WORK_DIR=/share/home/u19666033//syb/carla_garage/Bench2Drive
export PROJECT_ROOT=/share/home/u19666033//syb/carla_garage/

export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}/team_code
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

BASE_ROUTES=${WORK_DIR}/leaderboard/data/bench2drive220
ALGO=tfpp
PLANNER_TYPE=traj

# Check if the split_xml script needs to be executed
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    TASK_NUM=4
    python ${WORK_DIR}/tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi