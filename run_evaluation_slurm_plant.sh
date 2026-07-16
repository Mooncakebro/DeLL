#!/bin/bash
#SBATCH --job-name=eval_server
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/eval_server_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/eval_server_%a_%A.err
#SBATCH --partition=2080-galvani

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

for i in $(seq 1 1); do
  ex_name=$(printf "plant_000_%01d" "$((i - 1))")
  python -u evaluate_routes_slurm_plant.py --experiment "${ex_name}" --benchmark routes_validation --team_code team_code --epochs model_0046 --num_repetitions 3 &
done
wait

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
