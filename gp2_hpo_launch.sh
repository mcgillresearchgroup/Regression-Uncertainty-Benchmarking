#!/bin/sh
#SBATCH -J gp2_hpo
#SBATCH -o /lustre/home/frattarer/logs/gp2_hpo/hpo_%a.out
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-29

# 1 wrapper/model combo * 6 datasets * 3 train_percents (40, 60, 100%) = 30 jobs -> indices 0-29.
# Run `python main_cmd.py --list` first if you change MODEL_COMBINATIONS/DATASETS/TRAIN_PERCENTS,
# and update --array to match (0 to N-1).

export PYTHONPATH="/lustre/home/frattarer:$PYTHONPATH"
cd /lustre/home/frattarer/Benchmarking_2
python main_cmd.py \
    --job-index $SLURM_ARRAY_TASK_ID \
    --n-trials 100
