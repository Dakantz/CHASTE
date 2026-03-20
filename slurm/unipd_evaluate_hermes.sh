#!/bin/bash
#SBATCH --job-name=evaluate_clef_hermes
#SBATCH --array=0-23%4
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH --gres=gpu:a40 
#SBATCH -p allgroups
#SBATCH --output=logs/evaluate_%A_%a.out
#SBATCH --error=logs/evaluate_%A_%a.err
#SBATCH --time=4:00:00

# Array size is 48 to cover all combinations of:
# - 2 model types (3B, 8B)
# - 3 annotation types (base, entities, relations)
# - RAG vs no RAG
# # - gen-tokens 512 vs 2048
# - +naive or not



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
    "relations"
    "base"
)


FLAGS=""
out_file="eval_hermes"
# RAG vs no RAG
if [ $(($SLURM_ARRAY_TASK_ID%2)) -eq 0 ]; then
    FLAGS="$FLAGS --add-rag"
    echo "Using --add-rag"
    out_file="$out_file-rag"
else
    echo "Not using RAG"
fi
# model type
model_type=${model_types[($SLURM_ARRAY_TASK_ID/2)%2]}
annotation_type=${annotation_types[($SLURM_ARRAY_TASK_ID/4)%3]}
annotation_model_postfix=""
if [ "$annotation_type" != "base" ]; then
    annotation_model_postfix="-lora-$annotation_type"
    out_file="$out_file-lora-$annotation_type"
fi
FLAGS="$FLAGS --type $annotation_type"
FLAGS="$FLAGS --model-spec $quant_folder/$model_type$annotation_model_postfix.gguf"

echo "Annotation type: $annotation_type"
echo "Model type: $model_type"

echo "Using model $quant_folder/$model_type$annotation_model_postfix.gguf"
out_file="$out_file-$model_type-$annotation_type"

if [ $(($SLURM_ARRAY_TASK_ID/12)) -eq 0 ]; then
    FLAGS="$FLAGS --add-naive"
    echo "Using --add-naive" 
    out_file="$out_file-naive" 
else
    echo "No --add-naive"
fi
out_file="$out_file.json"

echo "Running with $FLAGS to $out_file"

python inference.py --model-provider llama --out-file $out_file $FLAGS 