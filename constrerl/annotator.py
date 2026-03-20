from html import entities
from itertools import count
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .erl_schema import (
    clean_label,
    entity_labels,
    relations as relation_labels,
)
from .annotation_model import (
    FullRelation,
    Metadata,
    Entity,
    AnnotatedArticle,
    Relation,
)
from llama_cpp import Llama, ChatCompletionRequestMessage, LlamaGrammar
from tqdm import tqdm
import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import pandas as pd

from sentence_transformers import SentenceTransformer
import torch as th
import numpy as np
from pydantic import BaseModel
import re
from sklearn.metrics.pairwise import cosine_similarity

from typing import TypeVar, Generic

from .sentences import (
    Sentence,
    AnnotationSpan,
    article_to_sentences,
    extract_noun_phrases,
    annotated_sentences_to_article,
)
from .concepts import ConceptDefinition

import os

ANNOTATION_SYSTEM_PROMPT = (
    """You are a medical expert annotating a medical scientific title and abstract."""
)

from abc import ABC, abstractmethod


class AnnotatorHelper:
    def __init__(
        self,
        model: Llama = None,
        langchain: BaseChatModel = None,
        gen_tokens=4096,
        system_prompt=ANNOTATION_SYSTEM_PROMPT,
        embedding_model="NeuML/pubmedbert-base-embeddings",
        top_k=10,
        add_few_shot=False,
        add_rag=False,
        reorder=False,
        add_entity_labels=False,
        naive_annotations=False,
        only_naive_annotations=False,
        score_reweights={
            "platinum": 1.0,
            "gold": 0.9,
            "silver": 0.8,
            "bronze": 0.7,
        },
    ):
        self.model = model
        self.relation_model = model
        self.entities_model = model
        self.langchain = langchain
        self.gen_tokens = gen_tokens
        if add_entity_labels:
            system_prompt = (
                system_prompt
                + " The possible entities are:\n"
                + "\n".join(
                    [f"{clean_label(e['label'])}: {e['desc']}" for e in entity_labels]
                )
            )
        self.system_message: list[ChatCompletionRequestMessage] = [
            {"role": "system", "content": system_prompt}
        ]
        self.example_messages = [*self.system_message]

        self.embedding_model = SentenceTransformer(embedding_model).to(
            "cuda" if th.cuda.is_available() else "mps"
        )

        self.top_k = top_k
        self.few_shot = add_few_shot
        self.naive_annotations = naive_annotations
        self.only_naive_annotations = only_naive_annotations
        self.rag = add_rag
        self.score_reweights = score_reweights
        self.reorder = reorder

        self.loaded_articles: dict[str, AnnotatedArticle] = {}
        self.loaded_sentences: dict[str, Sentence] = {}
        self.embeddings: dict[str, th.Tensor] = {}
        self.embeddings_sentences: dict[str, th.Tensor] = {}
        self.concepts: dict[str, ConceptDefinition] = {}

    @classmethod
    def relations_to_str(self, relations: list[Relation], sep="\n", sep_rel="|"):
        return sep.join(
            [
                sep_rel.join(
                    [
                        f"{rel.subject_label} ({rel.subject_text_span})",
                        rel.predicate,
                        f"{rel.object_label} ({rel.object_text_span})",
                    ]
                )
                for rel in relations
            ]
        )

    @classmethod
    def entities_to_str(self, entities: list[Entity], sep="\n"):
        return sep.join([f"{ent.label} ({ent.text_span})" for ent in entities])

    def embed_article(self, article: AnnotatedArticle):
        search_embedding = self.embedding_model.encode(
            [article.title + "\n" + article.abstract]
        )
        return search_embedding

    def load_articles(self, articles: dict[str, AnnotatedArticle]):
        for id, article in tqdm(list(articles.items()), desc="Embedding articles"):
            self.embeddings[id] = self.embed_article(article.metadata)
            sentences = article_to_sentences(
                article.metadata, article.relations, article.entities
            )
            sentence_texts = [sentence.text for sentence in sentences]
            embedded_sentences = self.embedding_model.encode(sentence_texts)
            for i, sentence in enumerate(sentences):
                sid = f"{id}_{sentence.start_idx}"
                self.loaded_sentences[sid] = sentence
                self.embeddings_sentences[sid] = embedded_sentences[i, :]
        self.loaded_articles = articles

    def load_articles_from_path(self, path: Path):
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "r") as f:
            data = json.load(f)
        for id, article in data["articles"].items():
            self.loaded_articles[id] = AnnotatedArticle.model_validate(article)
        for id, embedding in data["article_embeddings"].items():
            self.embeddings[id] = np.array(embedding)
        for sid, sentence in data["sentences"].items():
            self.loaded_sentences[sid] = Sentence.model_validate(sentence)
        for sid, sentence_embedding in data["sentence_embeddings"].items():
            self.embeddings_sentences[sid] = np.array(sentence_embedding)

    def save_articles(self, path: Path):
        with open(path, "w") as f:
            json.dump(
                {
                    "sentence_embeddings": {
                        sid: self.embeddings_sentences[sid].tolist()
                        for sid in self.embeddings_sentences.keys()
                    },
                    "article_embeddings": {
                        id: self.embeddings[id].tolist()
                        for id in self.embeddings.keys()
                    },
                    "articles": {
                        id: article.model_dump()
                        for id, article in self.loaded_articles.items()
                    },
                    "sentences": {
                        sid: sentence.model_dump()
                        for sid, sentence in self.loaded_sentences.items()
                    },
                },
                f,
            )

    def load_concepts(
        self, path: Path = Path("./data/Annotations/uri_collection_concepts.json")
    ):
        with open(path, "r") as f:
            data = json.load(f)
        self.concepts = {}
        if isinstance(data, list):
            for item in data:
                self.concepts[item["uri"]] = ConceptDefinition.model_validate(item)
        else:
            concepts_list = []
            for concept, dt in data.items():
                info = [
                    i
                    for i in dt["names"]
                    if i is not None and i not in ["names", "definitions"]
                ]
                # find all entities linked to this concept and find its type
                label = None
                counter = 0
                for art in self.loaded_articles.values():
                    for entity in art.entities or []:
                        if entity.uri == concept:
                            label = entity.label
                            counter += 1

                concepts_list.append(
                    {
                        "uri": concept,
                        "names_doc": " ".join(info),
                        "names": dt["names"] if dt["names"] else [],
                        "type": label,
                        "cnt": counter,
                    }
                )
                concepts = pd.DataFrame(concepts_list)
                concepts.to_json(
                    Path("./data/Annotations/uri_collection_concepts.json"),
                    orient="records",
                )
            for item in concepts_list:
                self.concepts[item["uri"]] = ConceptDefinition.model_validate(item)

    def __message_to_langchain(self, message: ChatCompletionRequestMessage):
        if message["role"] == "system":
            return SystemMessage(message["content"])
        if message["role"] == "user":
            return HumanMessage(message["content"])
        if message["role"] == "assistant":
            return AIMessage(message["content"])

    def find_similar_sentences(
        self, txt: str, id: str | None, ensure_relations_entities=False
    ) -> list[Sentence]:
        # search_embedding = self.embedding_model.encode(
        #     [article.title + "\n" + article.abstract],
        #     batch_size=12,
        #     max_length=8192,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
        # )["dense_vecs"]
        if id in self.embeddings_sentences.keys():
            search_embedding = self.embeddings_sentences[id]
        else:
            search_embedding = self.embedding_model.encode([txt])[0]
        # search_embedding = search_embedding[0]
        ids = [
            k
            for k in self.embeddings_sentences.keys()
            if id is None or (not k.startswith(id))
        ]
        if ensure_relations_entities:
            ids = [
                k
                for k in ids
                if (
                    ensure_relations_entities
                    and len(self.loaded_sentences[k].relations or []) > 0
                )
                or (len(self.loaded_sentences[k].entities or []) > 0)
            ]
        dense_matrix = np.stack([self.embeddings_sentences[id] for id in ids])
        similarities = dense_matrix @ search_embedding
        max_idx = np.argsort(similarities, axis=0)[-self.top_k :][::-1]
        best_matches_sentences = [self.loaded_sentences[ids[idx]] for idx in max_idx]
        return best_matches_sentences

    def annotate(
        self,
        articles: dict[str, Metadata],
        annotate: list[str] = [
            "entities",
            "relations",
        ],
    ) -> dict[str, AnnotatedArticle]:
        annotated_articles: dict[str, AnnotatedArticle] = {}
        annotators: list[Annotator] = []
        if "entities" in annotate:
            annotators.append(EntityAnnotator(self, self.entities_model))
        if "relations" in annotate:
            annotators.append(RelationAnnotator(self, self.relation_model))
        for annotator in annotators:
            annotated_articles_by_annotator = annotator.annotate(articles)
            if isinstance(annotator, EntityAnnotator):
                for id, article in annotated_articles_by_annotator.items():
                    if id not in annotated_articles:
                        annotated_articles[id] = article
                    else:
                        annotated_articles[id].entities = article.entities
            elif isinstance(annotator, RelationAnnotator):
                for id, article in annotated_articles_by_annotator.items():
                    if id not in annotated_articles:
                        annotated_articles[id] = article
                    else:
                        annotated_articles[id].relations = article.relations
        return annotated_articles

    def add_concept_uris(
        self,
        annotated_articles: dict[str, AnnotatedArticle],
        definitions_file=Path("./data/Annotations/merged_uri_definitions.json"),
    ):

        with open(definitions_file, "r") as f:
            uri_collection_definitions = json.load(f)

        concepts_list = []
        for concept, info in uri_collection_definitions.items():
            info = [
                i for i in info if i is not None and i not in ["names", "definitions"]
            ]
            # find all entities linked to this concept and find its type
            label = None
            for art in self.loaded_articles.values():
                for entity in art.entities or []:
                    if entity.uri == concept:
                        label = entity.label
                        break
                if label is not None:
                    break

            concepts_list.append(
                {"concept": concept, "names": " ".join(info), "type": label}
            )
        concepts_df = pd.DataFrame(concepts_list)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            strip_accents="ascii",
            ngram_range=(1, 2),
        )
        concept_vectors = vectorizer.fit_transform(concepts_df["names"])

        def find_best_uri_for_text(t: str, label: str | None) -> str | None:
            text_vector = vectorizer.transform([t])
            filter = (
                (concepts_df["type"] == label) | (concepts_df["type"].isna())
            ).to_numpy()
            filtered_concepts = concepts_df[filter]
            filtered_concept_vectors = concept_vectors[filter]
            similarities = cosine_similarity(text_vector, filtered_concept_vectors)
            n_candidates = 5
            best_idx = np.argsort(similarities[0])[-n_candidates:None]
            print(
                f"--- Finding concept for text: '{t}' with label: '{label}'. Top {n_candidates} candidates:"
            )
            for idx in best_idx:
                print(
                    f" -- Candidate concept: {filtered_concepts.iloc[idx]['concept']}, similarity: {similarities[0][idx]}, type: {filtered_concepts.iloc[idx]['names']}, concept type: {filtered_concepts.iloc[idx]['type']}, text: {t}, label: {label}"
                )

            best_concept = filtered_concepts.iloc[best_idx[-1]]["concept"]
            return best_concept

        for id, article in annotated_articles.items():
            for ent in article.entities or []:
                if ent.uri is not None:
                    continue
                ent.uri = find_best_uri_for_text(ent.text_span, ent.label)
            for rel in article.relations or []:
                if rel.subject_uri is not None:
                    continue
                rel.subject_uri = find_best_uri_for_text(
                    rel.subject_text_span, rel.subject_label
                )
                rel.object_uri = find_best_uri_for_text(
                    rel.object_text_span, rel.object_label
                )


T = TypeVar("T")


class Annotator(ABC, Generic[T]):
    def __init__(self, helper: AnnotatorHelper, model: Llama = None):
        self.helper = helper
        if model is not None:
            self.model = model
        else:
            self.model = helper.model

    @abstractmethod
    def apply_annotations(self, sentence: Sentence, annotations: list[T]):
        pass

    def annotate(self, articles: dict[str, Metadata]) -> dict[str, AnnotatedArticle]:
        annotated_articles = {}
        progress = tqdm(articles.items(), desc="Annotating articles")
        for id, article in progress:
            sentences = article_to_sentences(article)

            for sentence in sentences:
                annotations: list[T] = []
                if self.helper.naive_annotations:
                    annotations.extend(self.add_naive_annotations(sentence))
                if not self.helper.only_naive_annotations:
                    sentence_id = f"{id}_{sentence.start_idx}"
                    if len(sentence.text.strip()) == 0:
                        continue
                    phrases = extract_noun_phrases(sentence.text)
                    prompts = [*self.helper.system_message]
                    if self.helper.few_shot:
                        for ex in self.helper.example_messages:
                            prompts.extend(ex)
                    if self.helper.rag and len(self.helper.loaded_articles) > 0:
                        similar_sentences = self.helper.find_similar_sentences(
                            sentence.text, sentence_id, ensure_relations_entities=True
                        )
                        similar_sentence_messages = [
                            self.prompt_and_response(similar_sentence)
                            for similar_sentence in similar_sentences
                        ]
                        for ex in similar_sentence_messages:
                            prompts.extend(ex)

                    messages = prompts + [self.__prompt_sentence(sentence)]
                    chat_response = self.model.create_chat_completion(
                        messages,
                        max_tokens=self.helper.gen_tokens,
                        grammar=self.grammar(phrases),
                    )
                    response = chat_response["choices"][-1]["message"]["content"]
                    structure = self.response_to_structure(response, sentence, phrases)
                    annotations.extend(structure)
                self.apply_annotations(sentence, annotations)
            annotated_article = annotated_sentences_to_article(sentences, article)
            annotated_articles[id] = annotated_article
            progress.set_postfix({"id": id})
        return annotated_articles

    def __prompt_article(self, metadata: Metadata) -> ChatCompletionRequestMessage:
        return {
            "role": "user",
            "content": f"{metadata.title}\n{metadata.abstract}",
        }

    def __prompt_sentence(self, sent: Sentence) -> ChatCompletionRequestMessage:

        return {
            "role": "user",
            "content": sent.text,
        }

    @abstractmethod
    def response_to_structure(
        self, response: str, sent: Sentence, phrases: dict[str, AnnotationSpan]
    ) -> list[T]:
        pass

    @abstractmethod
    def sentence_to_response(self, structure: Sentence) -> str:
        pass

    @abstractmethod
    def grammar(self, phrases: dict[str, AnnotationSpan]) -> LlamaGrammar:
        pass

    def add_naive_annotations(self, sentence: Sentence) -> list[T]:
        return []

    def prompt_and_response(self, sent: Sentence) -> list[ChatCompletionRequestMessage]:
        return [
            self.__prompt_sentence(sent),
            {
                "role": "assistant",
                "content": self.sentence_to_response(sent),
            },
        ]

    def text_span_to_idxes(
        self, txt: str, phrases: dict[str, AnnotationSpan], span: str
    ) -> tuple[int, int] | None:
        if txt in phrases:
            # text_span = phrases[txt].text
            start_idx = phrases[txt].start_idx
            end_idx = phrases[txt].end_idx
        else:
            start_idx = span.lower().find(txt.lower())
            end_idx = start_idx + len(txt)
        if start_idx == -1:
            print(
                "WARNING: Could not find text span '{txt}' in sentence. This can lead to incorrect annotations."
                f"Sentence: {span}, text span: {txt}"
            )
            return None
        # the end_idx is exclusive, but the ground truth annotations are inclusive, so we subtract 1 from the end_idx
        return start_idx, end_idx - 1


class EntityAnnotator(Annotator[Entity]):
    def __init__(self, helper: AnnotatorHelper, model: Llama = None):
        super().__init__(helper, model)

    def apply_annotations(self, sentence, annotations):
        sentence.entities = annotations

    def __prompt_article(self, metadata: Metadata) -> ChatCompletionRequestMessage:
        return {
            "role": "user",
            "content": f"{metadata.title}\n{metadata.abstract}",
        }

    def add_naive_annotations(
        self, sentence: Sentence, do_filtering: bool = False
    ) -> list[Entity]:
        # go through all concept, and find direct string matches in the sentence, and add them as entities with the concept type as label
        annotated_entities: list[tuple[int, Entity]] = []

        for concept in self.helper.concepts.values():
            for name in concept.names:
                if name in sentence.text:
                    if len(name.strip()) <= 2:
                        continue
                    all_matches = [
                        m for m in re.finditer(rf"{re.escape(name)}", sentence.text)
                    ]
                    for match in all_matches:
                        start_idx = match.start()
                        end_idx = match.end() - 1
                        annotated_entities.append(
                            (
                                concept.cnt,
                                Entity(
                                    label=concept.type
                                    or "DDF",  # default to DDF if no type is given
                                    text_span=name,
                                    start_idx=start_idx,
                                    end_idx=end_idx,
                                    uri=concept.uri,
                                    location="title" if sentence.title else "abstract",
                                ),
                            )
                        )
        # find 'sub-entities' that are fully contained in another entity and remove them
        filtered_entities: list[Entity] = []
        if do_filtering:
            for e in annotated_entities:
                cnt, ent = e  # Unpack the count and entity
                # Check if this entity is fully contained in any other entity
                # and if the count is less than the other entity's count (to keep the most frequent one in case of ties)
                if any(
                    ent.start_idx >= other_ent.start_idx
                    and ent.end_idx <= other_ent.end_idx
                    # and cnt < other_cnt
                    for other_cnt, other_ent in annotated_entities
                    if other_ent != ent
                ):
                    continue
                filtered_entities.append(ent)
        else:
            filtered_entities = [ent for _, ent in annotated_entities]
        return filtered_entities

    def sentence_to_response(self, sent: Sentence) -> str:
        return "\n".join([f"{ent.label} ({ent.text_span})" for ent in sent.entities])

    def response_to_structure(
        self, response: str, sent: Sentence, phrases: dict[str, AnnotationSpan]
    ) -> list[Entity]:
        resolved_entities = []
        for line in response.split("\n"):
            match = re.match(r"(.+?)\s*\((.+)\)", line)
            if match:
                label = clean_label(match.group(1))
                text_span_raw = match.group(2).strip()
                start_idx, end_idx = self.text_span_to_idxes(
                    text_span_raw, phrases, sent.text
                ) or (
                    0,
                    len(sent.text) - 1,
                )
                text_span = (
                    sent.text[start_idx : end_idx + 1]
                    if start_idx is not None
                    else text_span_raw
                )
                resolved_entities.append(
                    Entity(
                        label=label,
                        text_span=text_span,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        location="title" if sent.title else "abstract",
                    )
                )
        return resolved_entities

    def grammar(self, phrases: dict[str, AnnotationSpan]) -> LlamaGrammar:
        resolved_entities = [lbl["label"] for lbl in entity_labels]
        entity_type_grammar = "|".join([f'"{e}"' for e in resolved_entities])
        entity_str_grammar = "|".join([f'"{n}"' for n in phrases.keys()])
        if len(entity_str_grammar) == 0:
            entity_str_grammar = "arbitrary-str"
        grammar_ebnf_str = rf"""root ::= ent-list
entity ::= entity-type" ("entity-str")"
arbitrary-str ::= (([0-9a-fA-F]|" "){{1, 4}})
entity-type ::= {entity_type_grammar}
entity-str ::= {entity_str_grammar}
ent-list ::= entity ([\n] entity)*
        """
        return LlamaGrammar(_grammar=grammar_ebnf_str)


class RelationAnnotator(Annotator):
    def __init__(self, helper: AnnotatorHelper, model: Llama = None):
        super().__init__(helper, model)

    def apply_annotations(self, sentence, annotations):
        sentence.relations = annotations

    def grammar(self, phrases: dict[str, AnnotationSpan]) -> LlamaGrammar:
        relationships = []
        for rel in relation_labels:
            for subj in rel["heads"]:
                for predicate in rel["predicate"]:
                    for obj in rel["tails"]:
                        relationships.append(
                            {
                                "subject_label": subj,
                                "predicate": predicate,
                                "object_label": obj,
                            }
                        )
        relationships_grammars = []
        for rel in relationships:
            relationships_grammars.append(
                f'"{rel["subject_label"]} ("entity-str") | {rel["predicate"]} | {rel["object_label"]} ("entity-str")"'
            )
        relationship_grammar = "|".join(relationships_grammars)
        entity_str_grammar = "|".join([f'"{n}"' for n in phrases.keys()])
        if len(entity_str_grammar) == 0:
            entity_str_grammar = "arbitrary-str"
        grammar_ebnf_str = rf"""
root ::= relationship-list
relationship-list ::= relationship ([\n] relationship)*
arbitrary-str ::= (([0-9a-fA-F]|" "){{1, 4}})
relationship ::= {relationship_grammar}
entity-str ::= {entity_str_grammar}
        """
        return LlamaGrammar(_grammar=grammar_ebnf_str)

    def sentence_to_response(self, sent: Sentence) -> str:
        return "\n".join(
            [
                "|".join(
                    [
                        f"{rel.subject_label} ({rel.subject_text_span})",
                        rel.predicate,
                        f"{rel.object_label} ({rel.object_text_span})",
                    ]
                )
                for rel in sent.relations
            ]
        )

    def response_to_structure(
        self, response: str, sent: Sentence, phrases: dict[str, AnnotationSpan]
    ) -> list[Relation]:
        relations = []
        for line in response.split("\n"):
            parts = line.split("|")
            if len(parts) == 3:
                subject_part, predicate, object_part = parts
                subject_match = re.match(r"(.+?)\s*\((.+)\)", subject_part.strip())
                object_match = re.match(r"(.+?)\s*\((.+)\)", object_part.strip())
                if subject_match and object_match:
                    subject_label = clean_label(subject_match.group(1))
                    subject_text_span_raw = subject_match.group(2).strip()
                    subject_start_idx, subject_end_idx = self.text_span_to_idxes(
                        subject_text_span_raw, phrases, sent.text
                    ) or (0, len(sent.text) - 1)
                    subject_text_span = sent.text[
                        subject_start_idx : subject_end_idx + 1
                    ]

                    object_label = clean_label(object_match.group(1))
                    object_text_span_raw = object_match.group(2).strip()
                    object_start_idx, object_end_idx = self.text_span_to_idxes(
                        object_text_span_raw, phrases, sent.text
                    ) or (0, len(sent.text) - 1)
                    object_text_span = sent.text[object_start_idx : object_end_idx + 1]

                    relations.append(
                        Relation(
                            subject_label=subject_label,
                            subject_text_span=subject_text_span,
                            subject_start_idx=subject_start_idx,
                            subject_end_idx=subject_end_idx,
                            subject_location="title" if sent.title else "abstract",
                            predicate=clean_label(predicate),
                            object_label=object_label,
                            object_text_span=object_text_span,
                            object_start_idx=object_start_idx,
                            object_end_idx=object_end_idx,
                            object_location="title" if sent.title else "abstract",
                        )
                    )
        return relations


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
