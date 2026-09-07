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
from typing import Any, TypeVar
import json
from pathlib import Path
import pandas as pd


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
    model_name = "3.2 3B" if "hermes-3-2-3B" in name or "323B" in name else "3.1 8B"
    if "baseline" in name.lower():
        model_name = " Baseline"
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
        graphwise_name = (
            rf"""\parbox{{2cm}}{{\vspace*{{0.3em}}{graphwise_name}\vspace*{{1em}}}}"""
        )
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
        "NED FT": "neefinetuned" in name,
        "Beams": beam_type(name),
        "Naive": "naive" in name,
        "Filter": "filtered" in name,
        "RAG": "rag" in name,
        # "Long $t$": "long" in name ,
        "LoRA": is_lora,
        # "fname": name,
    }

    if merge_mode:
        result_dict["Graphwise"] = graphwise_name if graphwise_name is not None else ""
        result_dict.pop("Filter", None)
        result_dict.pop("Naive", None)
        result_dict.pop("RAG", None)
    if k is not None:
        result_dict["$k$"] = k
    set_op = "$\cup$" if "union" in name else "$\cap$" if "intersection" in name else ""
    if merge_mode:
        result_dict["Set"] = set_op
    for k, v in result_dict.items():
        if isinstance(v, bool):
            result_dict[k] = tf[v] if model_name != " Baseline" else "-"
    return result_dict


def calculate_improvements(
    df_in: pd.DataFrame, base: dict[str, Any] = {"Beams": "$\\times$"}
) -> pd.DataFrame:
    index_cols = list(df_in.index.names)
    val_cols = list(df_in.columns)
    df = df_in.copy().reset_index()
    check_keys = [ck for ck in index_cols if ck in base.keys()]
    other_keys = [k for k in index_cols if k not in check_keys]
    if len(check_keys) == 0:
        print(
            f"No base keys found in DataFrame columns: {df.columns.tolist()}. Returning original DataFrame."
        )
        return df
    options = []
    for ck in check_keys:
        other_options = [o for o in df[ck].unique().tolist() if o != base[ck]]
        options.append((ck, other_options))
    print(
        f"Calculating improvements based on keys: {check_keys} with options: {options} / other keys: {other_keys}"
    )
    improveds: list[dict[str, Any]] = []
    for ck, other_options in options:
        for other_option in other_options:
            base_rows = df[df[ck] == base[ck]]
            other_rows = df[df[ck] == other_option]
            # match rows
            for i, base_row in base_rows.iterrows():
                matching_others = other_rows[
                    (other_rows[other_keys] == base_row[other_keys]).all(axis=1)
                ]
                if len(matching_others) == 0:
                    # print(
                    #     f"No matching row found for base row with {other_keys}={base_row[other_keys].to_dict()} when comparing {ck}={base[ck]} to {ck}={other_option}."
                    # )
                    continue

                improvements = (matching_others[val_cols] - base_row[val_cols]).iloc[0]
                improveds.append(
                    improvements.to_dict()
                    | matching_others.iloc[0][index_cols].to_dict()
                )
    if len(improveds) == 0:
        print("No improvements calculated. Returning empty DataFrame.")
        return pd.DataFrame()
    print(f"Resetting {index_cols=}")
    return pd.DataFrame(improveds).set_index(index_cols).sort_index()


def df_topk(
    df: pd.DataFrame, k: int, metrics: list = ["$F_{1,micro}$"]
) -> pd.DataFrame:
    idx_cols = df.index.names
    val_cols = df.columns.tolist()

    df_cp = df.copy().reset_index()
    df_sorted = df_cp.sort_values(metrics, ascending=False)
    topk_df = df_sorted.head(k)
    topk_df.set_index(idx_cols, inplace=True)
    topk_df = topk_df.sort_index()
    return topk_df[val_cols]


def calculate_relative_improvements(
    df_in: pd.DataFrame,
    df_other: pd.DataFrame,
    relevant_metrics: list[str] = ["$F_1$", "$F_{1,micro}$"],
    top: int = 10,
) -> pd.DataFrame:
    index_cols = list(df_in.index.names)
    val_cols = list(df_in.columns)
    df = df_in.copy().reset_index()
    df = df.sort_values(relevant_metrics, ascending=False).head(top)
    df_other = df_other.copy().reset_index()
    improveds: list[dict[str, Any]] = []
    # match rows
    for i, base_row in df.iterrows():
        matching_others = df_other[
            (df_other[index_cols] == base_row[index_cols]).all(axis=1)
        ]
        if len(matching_others) == 0:
            # print(
            #     f"No matching row found for base row with {other_keys}={base_row[other_keys].to_dict()} when comparing {ck}={base[ck]} to {ck}={other_option}."
            # )
            continue

        improvements = (matching_others[val_cols] - base_row[val_cols]).iloc[0]
        improveds.append(
            {k: improvements.to_dict()[k] for k in relevant_metrics}
            | {
                f"Relative {k}": matching_others.iloc[0][k] / base_row[k] - 1
                if base_row[k] != 0
                else 0
                for k in relevant_metrics
            }
            | matching_others.iloc[0][index_cols].to_dict()
        )
    if len(improveds) == 0:
        print("No improvements calculated. Returning empty DataFrame.")
        return pd.DataFrame()
    return pd.DataFrame(improveds).set_index(index_cols).sort_index()


def task_id_to_name(task_id: str) -> str:
    task_map = {
        "6.1.1": "\\gls{ner}",
        "6.1.2": "\\gls{nerd}",
        "6.2.1": "\\gls{re}",
        "6.2.2": "\\gls{mre}",
    }
    return task_map.get(task_id, task_id)
