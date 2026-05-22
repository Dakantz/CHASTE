#!/bin/bash
#SBATCH --job-name=evaluate_clef_hermes
#SBATCH --array=24-71%2
#SBATCH -c 2
#SBATCH --mem 12G
#SBATCH --output=logs/evaluate_%A_%a.out
#SBATCH --error=logs/evaluate_%A_%a.err
#SBATCH --time=1-00:00:00

# Array size is 48 to cover all combinations of:
# - 2 model types (3B, 8B)
# - 3 annotation types (base, entities, relations)
# - RAG vs no RAG
# # - gen-tokens 512 vs 2048
# - +naive or not
# - 

bash tug_evaluate_hermes_dev.sh