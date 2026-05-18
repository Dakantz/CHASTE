import re

from .annotation_model import (
    AnnotatedArticle,
    Metadata,
    Entity,
    Relation,
    FullRelation,
)
from .annotator import AnnotationTypes
from pydantic import BaseModel
from typing import TypeVar
import json
from pathlib import Path


def load_train(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    articles: dict[str, AnnotatedArticle] = {}
    for id, article in data.items():
        articles[id] = AnnotatedArticle.model_validate(article)
    return articles


def load_test(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    articles: dict[str, Metadata] = {}
    for id, article in data.items():
        articles[id] = Metadata.model_validate(article)
    return articles


E = TypeVar("E", bound=BaseModel)


def unique_model(ents: list[E], model: type[E]) -> list[E]:

    unique_ents = set(ent.model_dump_json() for ent in ents)
    return [model.model_validate_json(ent) for ent in unique_ents]


class EvalData(BaseModel):
    mention_level_relations: list[Relation]
    concept_level_relations: list[Relation]
    entities: list[Entity]


def prepare_for_eval(articles: dict[str, AnnotatedArticle]):
    prepared = {}
    for id, article in articles.items():
        entities_unique = unique_model(article.entities or [], Entity)
        relations_unique = unique_model(article.relations or [], Relation)
        eval_data = EvalData(
            mention_level_relations=relations_unique,
            concept_level_relations=relations_unique,
            entities=entities_unique,
        )
        prepared[id] = eval_data.model_dump()
    return prepared


def extract_flags_from_name(
    name: str,
    tf: dict[bool, str] = {True: "\\checkmark", False: "$\\times$"},
    merge_mode: bool = False,
    test_mode: bool = False,
    k: int | None = None,
) -> dict[str, bool]:

    finetune_names = [
        AnnotationTypes.ENTITY,
        AnnotationTypes.RELATION,
    ]
    model_name = "3.2 1B" if "hermes-3-2-3B" in name or "323B" in name else "3.1 8B"
    graphwise_name = None

    def get_graphwise_name(name: str) -> str:
        splits = name.split("_")
        graphwise_idx = next(
            (i for i, s in enumerate(splits) if s.lower() == "graphwise"), None
        )
        if graphwise_idx is not None and graphwise_idx + 2 < len(splits):
            graphwise_name = "_".join(splits[graphwise_idx + 2 : -1])
        graphwise_name = graphwise_name.replace("union", "").replace("intersection", "")
        # graphwise_name = re.sub(r"T\d+", "", graphwise_name)
        graphwise_name = re.sub(r"(\d+[\_-])", "", graphwise_name)
        graphwise_name = re.sub(r"-DEV", "", graphwise_name)
        graphwise_name = re.sub(r"BGPT5520G", "", graphwise_name)
        graphwise_name = re.sub(r"\_", "-", graphwise_name)
        return graphwise_name.strip("_- ")

    if merge_mode and not test_mode:
        graphwise_name = get_graphwise_name(name)
    if merge_mode and test_mode:
        # look up in submission folder
        submission_folder = Path("staging").glob(f"*{name}*")
        submission_file = next(submission_folder, None)
        print(f"Looking for submission file for {name}, found {submission_file}")
        if submission_file is not None:
            meta_file = submission_file / f"{submission_file.name}.meta"
            with open(meta_file, "r") as f:
                meta_data = f.read()
            full_run_id = re.findall(r"RunID: (.+)", meta_data, re.MULTILINE)
            print(f"Found full run id {full_run_id} for {name}")
            graphwise_name = (
                get_graphwise_name(full_run_id[0]) if full_run_id else graphwise_name
            )

    if graphwise_name is not None:
        graphwise_name = rf"""\parbox{{2cm}}{{{graphwise_name}}}"""
    if "eval_naive" in name or name.startswith("naive"):
        model_name = "Naive"
    # splits = model_name.split(" ")
    # if len(splits) > 2:
    #     model_name = splits[0] + " " + "-".join(splits[1:])
    is_lora = "lora" in name

    def beam_type(name: str) -> str:
        if "beam-end" in name or "beamend" in name:
            return "End"
        elif "beam-shallow" in name or "beamshallow" in name:
            return "Shallow"
        else:
            return False

    result_dict = {
        "Model": model_name,
        "NE FT": "neefinetuned" in name,
        "Beams": beam_type(name),
        "Naive": "naive" in name,
        "Filter": "filtered" in name,
        "RAG": "rag" in name,
        # "Long $t$": "long" in name ,
        "LoRA": is_lora,
        # "fname": name,
    }

    if merge_mode and graphwise_name is not None:
        result_dict["Graphwise"] = graphwise_name
        result_dict.pop("Filter", None)
    if k is not None:
        result_dict["$k$"] = k
    set_op = "$\cup$" if "union" in name else "$\cap$"
    if merge_mode:
        result_dict["Set"] = set_op
    for k, v in result_dict.items():
        if isinstance(v, bool):
            result_dict[k] = tf[v]
    return result_dict
