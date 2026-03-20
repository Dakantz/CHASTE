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

nlp = spacy.load("en_core_web_trf")


class AnnotationSpan(BaseModel):
    start_idx: int
    end_idx: int
    text: str


def extract_noun_phrases(txt: str) -> dict[str, AnnotationSpan]:
    # Load the English NLP model

    # Parse the text
    doc = nlp(txt)

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
    return noun_phrases


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
