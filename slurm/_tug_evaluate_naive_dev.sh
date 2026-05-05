#!/bin/bash
#SBATCH --job-name=evaluate_clef_hermes
#SBATCH --array=0-1%2
#SBATCH -c 2
#SBATCH --mem 12G
#SBATCH --gres=gpu 
#SBATCH --account=bkantz
#SBATCH --output=logs/evaluate_%A_%a.out
#SBATCH --error=logs/evaluate_%A_%a.err
#SBATCH --time=1:00:00

# Array size is 1 to cover all combinations of:
# - filter vs no filter

bash tug_evaluate_naive_dev.sh