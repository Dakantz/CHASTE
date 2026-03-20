from .annotation_model import (
    AnnotatedArticle,
    Metadata,
    Entity,
    Relation,
    FullRelation,
)
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
