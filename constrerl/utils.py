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


def prepare_for_eval(articles: dict[str, AnnotatedArticle]):
    prepared = {}
    for id, article in articles.items():
        entities_unique = unique_model(article.entities or [], Entity)
        relations_unique = unique_model(article.relations or [], Relation)
        prepared[id] = {
            "mention_level_relations": [rel.model_dump() for rel in relations_unique],
            "concept_level_relations": [rel.model_dump() for rel in relations_unique],
            "entities": [ent.model_dump() for ent in entities_unique],
        }
    return prepared
