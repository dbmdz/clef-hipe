import click
import json
import sys

import flair
import torch

from typing import List

from flair.data import MultiCorpus
from flair.datasets import ColumnCorpus, NER_HIPE_2022
from flair.embeddings import (
    TokenEmbeddings,
    StackedEmbeddings,
    TransformerWordEmbeddings
)
from flair import set_seed
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

def run_evaluator(model_name: str, dataset_names: str):
    corpus_list = [] 

    # Dataset-related
    for dataset in dataset_names.split((",")):
        dataset_name, language = dataset.split("/")
        corpus_list.append(NER_HIPE_2022(dataset_name=dataset_name, language=language, add_document_separator=True))
    

    corpora: MultiCorpus = MultiCorpus(corpora=corpus_list, sample_missing_splits=False)
    label_dictionary = corpora.make_label_dictionary(label_type="ner")
    print("Label Dictionary:", label_dictionary.get_items())

    model = SequenceTagger.load(model_name)

    dev_result = model.evaluate(corpora.dev, gold_label_type="ner", mini_batch_size=8)

    print(dev_result)

if __name__ == "__main__":
    # Read JSON configuration
    model_name = sys.argv[1] # Usually ends with best_model.pt
    dataset_names = sys.argv[2] # Use , as dataset delimiter!

    run_evaluator(model_name, dataset_names)
