from pathlib import Path
from typing import List, Optional, Any
from pydantic import BaseModel
import json


class Conceptlevelrelation(BaseModel):
    subject_uri: str
    subject_label: str
    predicate: str
    object_uri: str
    object_label: str


class Mentionlevelrelation(BaseModel):
    subject_text_span: str
    subject_label: str
    predicate: str
    object_text_span: str
    object_label: str


class FullRelation(BaseModel):
    subject_uri: str
    subject_text_span: str
    subject_label: str
    predicate: str
    object_uri: str
    object_text_span: str
    object_label: str


class Relation(BaseModel):
    subject_start_idx: int
    subject_end_idx: int
    subject_location: str
    subject_text_span: str
    subject_label: str
    subject_uri: str | None = None
    predicate: str
    object_start_idx: int
    object_end_idx: int
    object_location: str
    object_text_span: str
    object_label: str
    object_uri: str | None = None


class Entity(BaseModel):
    start_idx: int
    end_idx: int
    location: str
    text_span: str
    label: str
    uri: str | None = None


class NEREntity(BaseModel):
    start_idx: int
    end_idx: int
    location: str
    text_span: str
    label: str


class NERDEntity(BaseModel):
    start_idx: int
    end_idx: int
    location: str
    text_span: str
    label: str
    uri: str


class Metadata(BaseModel):
    title: str
    abstract: str
    author: str | None | float = None
    journal: str | None = None
    year: int | None = None
    annotator: str | None = None


class AnnotatedArticle(BaseModel):
    metadata: Metadata
    entities: List[Entity] | None = None

    relations: List[Relation] | None = None

    mention_level_relations: List[Mentionlevelrelation] | None = None
    concept_level_relations: List[Conceptlevelrelation] | None = None


def get_gt_entity_labels_response(annotated_article: AnnotatedArticle) -> set[str]:
    entities = annotated_article.entities or []
    r_str = ",".join([e.label for e in entities])
    return r_str


def load_collection(
    collection: str = "Dev", root_data_path: Path = Path("./data/Annotations")
) -> dict[str, AnnotatedArticle]:
    jsons = (root_data_path / collection).rglob("*.json")
    annotated_articles: dict[str, AnnotatedArticle] = {}
    for j in jsons:
        with open(j, "r") as f:
            articles = json.load(f)
            for id, article in articles.items():
                try:
                    annotated_articles[id] = AnnotatedArticle.model_validate(article)
                except Exception as e:
                    print(f"Failed to load article {id} from file {j}: {e}")
                    raise e
    return annotated_articles
