# GutBrain IE Challenge @ CLEF 2026

`Benedikt Kantz, Peter Walder, Stefan Lengauer, Tobias Schreck`

## reloaded for '26 - I have become CHASTE, master of CLEANR

* `C`onstrained
* en`H`anced
* `A`nnotator
* u`S`ing
* on`T`ological
* `E`ntities


## Our appraoch

* extract noun phrases using space
* constrain LLM using simple (!) grammar
* Use finetuned HERMES 3.2 1B or 3B

## Setup

```sh
# Install dependencies
git submodule update
git submodule update --init --recursive
uv sync --prerelease=allow   
source .venv/bin/activate
cd llama-cpp-python
# if you want to use the GPU:
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_BUILD_PARALLEL_LEVEL=8" uv pip install -e . 
# if you want to use you metal processor:
CMAKE_ARGS="-DGGML_METAL=on"  uv pip install -e .
# on a cluster you could start into a interactive environment:
srun --gres=gpu:a40 -c 12 --partition allgroups  --time=10:00  --pty   bash

# dowload models (make sure to set you HF token!)
tune download NousResearch/Hermes-3-Llama-3.2-3B  --output-dir models/hermes-3-2-3B
huggingface-cli download meta-llama/Llama-3.2-3B original/tokenizer.model --local-dir  models/hermes-3-2-3B
tune download NousResearch/Hermes-3-Llama-3.1-8B  --output-dir models/hermes-3-1-8B
huggingface-cli download meta-llama/Llama-3.1-8B original/tokenizer.model --local-dir  models/hermes-3-1-8B
python manage_models/quantize_all.py

