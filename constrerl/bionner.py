from pathlib import Path
import pandas as pd
from .annotation_model import AnnotatedArticle, Relation, Entity, Metadata


def load_data(base_path=Path("NEREL-BIO/BioNNE-R/data/en/train/"), collection="train"):
    rel_data = pd.read_table(base_path / f"eng-{collection}-rel.tsv")
    ent_data = pd.read_table(base_path / f"eng-{collection}-ent.tsv")
    articles: dict[str, AnnotatedArticle] = {}
    article_ids = set(rel_data["document_id"]).union(set(ent_data["document_id"]))
    for article_id in article_ids:
        entities = ent_data[ent_data["document_id"] == article_id]
        entities_data: list[Entity] = []
        for _, row in entities.iterrows():
            entities_data.append(
                Entity(
                    start_idx=int(row["entity_span"].split("-")[0]),
                    end_idx=int(row["entity_span"].split("-")[1]),
                    label=row["entity_type"],
                    text_span=row["entity_text"],
                    location="abstract",
                )
            )
        relations = rel_data[rel_data["document_id"] == article_id]
        relations_data: list[Relation] = []
        for _, row in relations.iterrows():
            relations_data.append(
                Relation(
                    predicate=row["relation"],
                    subject_label=row["head_type"],
                    subject_text_span=row["head_text"],
                    subject_start_idx=int(row["head_span"].split("-")[0]),
                    subject_end_idx=int(row["head_span"].split("-")[1]),
                    object_label=row["tail_type"],
                    object_text_span=row["tail_text"],
                    object_start_idx=int(row["tail_span"].split("-")[0]),
                    object_end_idx=int(row["tail_span"].split("-")[1]),
                    subject_location="abstract",
                    object_location="abstract",
                )
            )
        text = ""
        with open(base_path / "texts" / f"{article_id}.txt", "r") as f:
            text = f.read()
        articles[article_id] = AnnotatedArticle(
            entities=entities_data,
            relations=relations_data,
            metadata=Metadata(
                abstract=text,
                title="",
            ),
        )
    return articles
