#!/bin/bash
#SBATCH --job-name=evaluate_clef_hermes
#SBATCH --array=0-23%4
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH --gres=gpu:rtx 
#SBATCH -p allgroups
#SBATCH --output=logs/evaluate_%A_%a.out
#SBATCH --error=logs/evaluate_%A_%a.err
#SBATCH --time=4:00:00

# Array size is 1 to cover all combinations of:
# - naive



cd ..
source .venv/bin/activate

# either use --add-rag or --reorder bases on $SLURM_ARRAY_TASK_ID

quant_folder="quants"

model_types=(
    "hermes-3-2-3B"
    "hermes-3-1-8B"
)

annotation_types=(
    "entities"
)


FLAGS=""
out_file="eval_naive"
annotation_type=${annotation_types[($SLURM_ARRAY_TASK_ID)%1]}
annotation_model_postfix=""

FLAGS="$FLAGS --type $annotation_type"

echo "Annotation type: $annotation_type"
out_file="$out_file.json"

echo "Running with $FLAGS to $out_file"

python inference.py --model-provider llama --out-file $out_file $FLAGS 