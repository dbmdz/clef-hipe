from pathlib import Path
from typing import List

import torch

import flair.datasets
from flair.data import Corpus
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    BytePairEmbeddings
)

# 1. get the corpus
corpus: Corpus = flair.datasets.ColumnCorpus(data_folder=Path("../../preprocessed-v1.2/de/"),
                                             train_file="train.txt",
                                             dev_file="dev.txt",
                                             test_file="dev.txt",
                                             column_format={0: "token", 1: "ner"},
                                             tag_to_bioes="ner",
                                             skip_first_line=True)
print(corpus)

# 2. what tag do we want to predict?
tag_type = "ner"

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    BytePairEmbeddings(language="de", dim=300, syllables=200000)
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
    use_crf=True,
)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(model=tagger, corpus=corpus, use_tensorboard=True)

trainer.train(
    "resources/taggers/baseline-de-bpemb-3",
    mini_batch_size=16,
    patience=5,
    max_epochs=200,
)
