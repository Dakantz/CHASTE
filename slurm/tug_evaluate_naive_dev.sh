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



cd ..
source .venv/bin/activate

# either use --add-rag or --reorder bases on $SLURM_ARRAY_TASK_ID

quant_folder="quants"
annotation_types=(
    "entities"
)
FLAGS="--add-naive --naive-only"
out_file="eval_naive"
annotation_type=${annotation_types[$SLURM_ARRAY_TASK_ID%1]}


if [ $(($SLURM_ARRAY_TASK_ID%2)) -eq 0 ]; then
    FLAGS="$FLAGS --naive-filter"
    echo "Using --naive-filter" 
    out_file="$out_file-filtered" 
else
    echo "No --naive-filter"
fi

out_file="$out_file-$annotation_type"

FLAGS="$FLAGS --type $annotation_type"

echo "Annotation type: $annotation_type"
out_file="$out_file.json"

echo "Running with $FLAGS to $out_file"

python inference.py --model-provider naive --out-file $out_file $FLAGS --data-path data/Annotations/prepared_train.json --eval-path data/Articles/json_format/articles_dev.json --out-path data/results_dev