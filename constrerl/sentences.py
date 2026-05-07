from abc import abstractmethod
import abc
from pathlib import Path
from pydantic import BaseModel
import re
import spacy

from constrerl.annotation_model import (
    Entity,
    FullRelation,
    Relation,
    Metadata,
    AnnotatedArticle,
)
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


class AnnotationSpan(BaseModel):
    start_idx: int
    end_idx: int
    text: str


class SentenceAnnotator(abc.ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract_noun_phrases(self, txt: str) -> dict[str, AnnotationSpan]:
        pass


class SpacyAnnotator(abc.ABC):
    def __init__(self, model_name: str = "en_core_web_trf"):
        self.nlp = spacy.load(model_name)

    def extract_noun_phrases(self, txt: str) -> dict[str, AnnotationSpan]:
        # Load the English NLP model

        # Parse the text
        doc = self.nlp(txt)

        # Extract noun phrases
        noun_phrases = {
            chunk.text: AnnotationSpan(
                start_idx=chunk.start_char,
                end_idx=chunk.end_char,
                text=chunk.text,
            )
            for chunk in doc.noun_chunks
        }
        # remove 'a ', 'an ', 'the ' from the beginning of noun phrases
        for k, np in (dict(noun_phrases)).items():
            new_text = re.sub(r"^(a|an|the)\s+", "", np.text, flags=re.IGNORECASE)
            # also remove any leading special characters
            new_text = re.sub(r"^[^\w]+", "", new_text)
            if new_text != np.text:
                noun_phrases[new_text] = AnnotationSpan(
                    start_idx=np.start_idx + len(np.text) - len(new_text),
                    end_idx=np.end_idx,
                    text=new_text,
                )
                noun_phrases.pop(k)

            new_text_end = re.sub(r"[\"\{\}]+", "", new_text)
            if new_text_end != new_text:
                noun_phrases[new_text_end] = AnnotationSpan(
                    start_idx=np.start_idx,
                    end_idx=np.end_idx,
                    text=new_text_end,
                )
                noun_phrases.pop(new_text)
        for p in noun_phrases.values():
            p.end_idx = p.end_idx - 1
        return noun_phrases


class BERTAnnotator(SentenceAnnotator):
    def __init__(self, model_path: Path = Path("./finetuned/ned/best_model/")):
        super().__init__()
        self.label_list = ["O", "B-NE", "I-NE"]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{model_path.absolute()}/",
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            f"{model_path.absolute()}/",
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def extract_noun_phrases(self, txt: str) -> dict[str, AnnotationSpan]:
        # Implement noun phrase extraction using your BERT model
        annotation_spans: dict[str, AnnotationSpan] = {}
        text = txt
        tokens = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        input_ids = (
            torch.tensor(tokens["input_ids"]).unsqueeze(0).to(self.model.device)
        )  # batch size 1
        attention_mask = (
            torch.tensor(tokens["attention_mask"]).unsqueeze(0).to(self.model.device)
        )
        predictions = self.model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_labels = predictions.logits.argmax(dim=-1).squeeze().tolist()
        offsets = tokens["offset_mapping"]
        spans: list[AnnotationSpan] = []
        current_span: AnnotationSpan = None
        last_end_idx = 0

        for label_id, (start, end) in zip(predicted_labels, offsets):
            if label_id == self.label2id["O"]:
                continue
            if label_id == self.label2id["B-NE"]:
                if current_span is not None:
                    current_span.end_idx = last_end_idx
                    current_span.text = text[
                        current_span.start_idx : current_span.end_idx
                    ]
                    spans.append(current_span)
                current_span = AnnotationSpan(
                    start_idx=start,
                    end_idx=end,
                    text=text[start:end],
                )
            last_end_idx = end
        if current_span is not None:
            current_span.end_idx = last_end_idx
            current_span.text = text[current_span.start_idx : current_span.end_idx]
            spans.append(current_span)
        filtered_ents = []
        for ant in spans:
            if ant.text.strip() != "":
                ant.end_idx = ant.end_idx - 1
                filtered_ents.append(ant)
                annotation_spans[ant.text] = AnnotationSpan(
                    start_idx=ant.start_idx,
                    end_idx=ant.end_idx,
                    text=ant.text,
                )
        return annotation_spans


class Sentence(BaseModel):
    start_idx: int
    from_article: Metadata
    text: str
    title: bool = False

    entities: list[Entity] | None = None
    relations: list[FullRelation | Relation] | None = None


def article_to_sentences(
    article: Metadata, relations: list[Relation] = [], entities: list[Entity] = []
):
    sentences: list[Sentence] = []

    title_entities = [ent for ent in entities if ent.location.lower() == "title"]
    title_relations = [
        rel
        for rel in relations
        if (rel.object_location.lower() == "title")
        or (rel.subject_location.lower() == "title")
    ]
    sentences.append(
        Sentence(
            start_idx=0,
            from_article=article,
            text=article.title,
            title=True,
            entities=title_entities,
            relations=title_relations,
        )
    )
    sentence_text = re.split(r"\.[\n ]+", article.abstract)  # A *very* basic heuristic
    last_idx = 0
    for sentence in sentence_text:
        start_idx = article.abstract.find(sentence, max(0, last_idx - 2))
        if start_idx == -1:
            raise ValueError(
                f"Sentence {sentence} not contained in abstract, start_idx {last_idx}."
            )
        end_idx = start_idx + len(sentence)
        sentence_entities: list[Entity] = [
            Entity.model_copy(ent)
            for ent in entities
            if ent.start_idx >= start_idx
            and ent.end_idx <= end_idx
            and ent.location.lower() == "abstract"
        ]
        for ent in sentence_entities:
            ent.start_idx = ent.start_idx - start_idx
            ent.end_idx = ent.end_idx - start_idx
        sentence_relations: list[Relation] = [
            Relation.model_copy(rel)
            for rel in relations
            if (
                (rel.object_start_idx >= start_idx and rel.object_end_idx <= end_idx)
                or (
                    rel.subject_start_idx >= start_idx
                    and rel.subject_end_idx <= end_idx
                )
            )
            and (
                rel.object_location.lower() == "abstract"
                or rel.subject_location.lower() == "abstract"
            )
        ]
        for rel in sentence_relations:
            rel.subject_start_idx = rel.subject_start_idx - start_idx
            rel.subject_end_idx = rel.subject_end_idx - start_idx
            rel.object_start_idx = rel.object_start_idx - start_idx
            rel.object_end_idx = rel.object_end_idx - start_idx
        sentences.append(
            Sentence(
                start_idx=start_idx,
                from_article=article,
                text=sentence,
                entities=sentence_entities,
                relations=sentence_relations,
            )
        )
        last_idx = end_idx
    return sentences


def annotated_sentences_to_article(
    sentences: list[Sentence], metadata: Metadata
) -> AnnotatedArticle:
    all_entities: list[Entity] = []
    all_relations: list[FullRelation] = []
    for sentence in sentences:
        if sentence.entities is not None:
            for sen in sentence.entities:
                sen.start_idx = sen.start_idx + sentence.start_idx
                sen.end_idx = sen.end_idx + sentence.start_idx
            all_entities.extend(sentence.entities)
        if sentence.relations is not None:
            for rel in sentence.relations:
                rel.subject_start_idx = rel.subject_start_idx + sentence.start_idx
                rel.subject_end_idx = rel.subject_end_idx + sentence.start_idx
                rel.object_start_idx = rel.object_start_idx + sentence.start_idx
                rel.object_end_idx = rel.object_end_idx + sentence.start_idx
            all_relations.extend(sentence.relations)
    return AnnotatedArticle(
        metadata=metadata,
        entities=all_entities,
        relations=all_relations,
    )
