# %%
from email.policy import default

from llama_cpp import Llama

from constrerl.annotator import AnnotatedArticle, AnnotatorHelper, AnnotationTypes
from constrerl.utils import prepare_for_eval

# %%
import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-provider", type=str, default="llama")
    parser.add_argument(
        "--model-spec", type=str, default="quants/llama-3-2-1B-instruct-lora.gguf"
    )
    parser.add_argument(
        "--data-path", type=str, default="data/Annotations/prepared_train.json"
    )
    parser.add_argument(
        "--eval-path", type=str, default="data/Annotations/Dev/json_format/dev.json"
    )
    parser.add_argument("--out-path", type=str, default="data/results_dev")
    parser.add_argument("--out-file", type=str, default="dev_out.json")
    parser.add_argument("--type", type=str, default="entities")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gen-tokens", type=int, default=512)
    parser.add_argument("--ctx", type=int, default=8196)
    parser.add_argument("--add-rag", default=False, action="store_true")
    parser.add_argument("--add-naive", default=False, action="store_true")
    parser.add_argument("--naive-filter", default=False, action="store_true")
    parser.add_argument("--naive-only", default=False, action="store_true")
    args = parser.parse_args()
    print("Starting with", args)
    model: Llama = None
    match args.model_provider:
        # case "openai":
        #     # llm = init_chat_model("ft:gpt-4o-mini-2024-07-18:tu-graz-hereditary:gutbrain-ie-finetune:B5qr9cGV", model_provider="openai")
        #     llm = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")

        case "llama":
            if args.model_spec.endswith(".gguf"):
                model_path = args.model_spec  # "quants/llama-3-2-1B-instruct-lora.gguf"
                model = Llama(
                    model_path,
                    n_gpu_layers=-1,
                    n_ctx=args.ctx,
                    temperature=0.1,
                    # draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10),
                )
            else:
                model = Llama.from_pretrained(
                    args.model_spec,
                    filename="*.Q8_0.gguf",
                    n_gpu_layers=-1,
                    n_ctx=args.ctx,
                    temperature=0.1,
                )
        case "naive":
            print("Using naive annotator, no model will be loaded")
        # default:
        case _:
            print("Unknown model provider", args.model_provider)
    data_path = args.data_path
    out_path = Path(args.out_path) / args.out_file
    annotator = AnnotatorHelper(
        model=model,
        gen_tokens=args.gen_tokens,
        add_rag=args.add_rag,
        naive_annotations=args.add_naive,
        naive_only=args.naive_only,
        naive_filter=args.naive_filter,
        top_k=args.top_k,
    )
    print("Loading articles from", data_path)
    annotator.load_articles_from_path(Path(data_path))
    print("-->> Loaded articles:", len(annotator.loaded_articles))
    print("Loading evaluation set from", args.eval_path)
    with open(args.eval_path, "r") as f:
        eval_set = json.load(f)
    eval_set = {
        id: AnnotatedArticle.model_validate(article) for id, article in eval_set.items()
    }
    print("-->> Loaded eval set articles:", len(eval_set))

    # %%

    annotations_types = (
        [AnnotationTypes.from_str(args.type)]
        if args.type in ["entities", "relations"]
        else [AnnotationTypes.ENTITY, AnnotationTypes.RELATION]
    )
    print("Annotating with types", annotations_types)
    annotations: dict[str, AnnotatedArticle] = annotator.annotate(
        {id: article.metadata for id, article in list(eval_set.items())},
        annotate=annotations_types,
    )
    annotator.add_concept_uris(annotations)

    output_data = prepare_for_eval(annotations)
    # %%
    with open(out_path, "w") as f:
        json.dump(output_data, f)
    # %%
    print("Done")
