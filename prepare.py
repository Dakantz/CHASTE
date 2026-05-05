from constrerl.annotator import (
    Annotator,
    RelationAnnotator,
    EntityAnnotator,
    AnnotatorHelper,
    article_to_sentences,
)
from constrerl.annotation_model import (
    AnnotatedArticle,
    load_collection,
)
from tqdm import tqdm
from pathlib import Path

collections = ["Train"]
train_data: dict[str, AnnotatedArticle] = {}
for collection in collections:
    train_data.update(load_collection(collection))

annotator_helper = AnnotatorHelper(top_k=5)
annotator_helper.load_articles(train_data)


out_path = Path("./data/Annotations/prepared_train.json")
annotator_helper.save_articles(out_path)
