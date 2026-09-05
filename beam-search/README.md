# Beam Search over Efficient `llama.cpp` generations

## Theory

> Forthcoming...

## Example Usage

```python
from llama_cpp_beamsearch.completion import BeamSearchCompletion
from llama_cpp_beamsearch.config import BeamSearchConfig
from llama_cpp import Llama, LlamaGrammar, ChatCompletionRequestMessage

model = Llama.from_pretrained(
    repo_id="unsloth/Qwen3-0.6B-GGUF",
    filename="*Q4_0.gguf",
    verbose=False,
    logits_all=True,
)
allowed_tokens = ["Wrench", "Screwdriver", "Ornament"]
allowed_tokens_str = " | ".join(f'"{t}"' for t in allowed_tokens)
grammar = LlamaGrammar(
    _grammar=f"""
    root ::= allowed_tokens
    start ::= ( allowed_tokens " "  )*
    allowed_tokens ::= {allowed_tokens_str}
    """
)
config = BeamSearchConfig(
    max_depth=1,
    end_token=None,
    k_progress=[2, 3],
)
completion = BeamSearchCompletion(model, grammar, config)

message = "Hello?"

result = completion.completion_beam_search(message, max_tokens=5)
assert re.match(r"^((Wrench|Screwdriver|Ornament) ?)*$", result)
```