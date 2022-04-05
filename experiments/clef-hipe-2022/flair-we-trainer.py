import click
import json
import sys

import flair
import torch

from typing import List

from flair.data import MultiCorpus
from flair.datasets import ColumnCorpus, NER_HIPE_2022
from flair.embeddings import (
    FastTextEmbeddings
)
from flair import set_seed
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

def run_experiment(seed, batch_size, epoch, learning_rate, hipe_datasets, json_config):
    # Config values
    # Replace it with more Pythonic solutions later!
    word_embedding = json_config["word_embedding"]
    use_crf = json_config["use_crf"] if "use_crf" in json_config else False

    # Set seed for reproducibility
    set_seed(seed)

    corpus_list = [] 

    # Dataset-related
    for dataset in hipe_datasets:
        dataset_name, language = dataset.split("/")
        corpus_list.append(NER_HIPE_2022(dataset_name=dataset_name, language=language, add_document_separator=True))

    print("Use CRF:", use_crf)

    corpora: MultiCorpus = MultiCorpus(corpora=corpus_list, sample_missing_splits=False)
    label_dictionary = corpora.make_label_dictionary(label_type="ner")
    print("Label Dictionary:", label_dictionary.get_items())

    # FastText Embeddings
    embeddings = FastTextEmbeddings(embeddings=word_embedding)

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dictionary,
        tag_type="ner",
        use_crf=use_crf,
    )

    # Trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpora)

    datasets = "-".join([dataset for dataset in hipe_datasets])

    trainer.train(
        f"hipe2022-flert-we-trainer-{datasets}-{word_embedding}-bs{batch_size}-wsFalse-e{epoch}-lr{learning_rate}-crf{use_crf}-{seed}",
        mini_batch_size=batch_size,
        mini_batch_chunk_size=2,
        patience=3,
        max_epochs=epoch,
        shuffle=True,
        learning_rate=learning_rate,
    )
    
    # Finally, print model card for information
    tagger.print_model_card()


if __name__ == "__main__":
    # Read JSON configuration
    filename = sys.argv[1]
    with open(filename, "rt") as f_p:
        json_config = json.load(f_p)

    seeds = json_config["seeds"]
    batch_sizes = json_config["batch_sizes"]
    epochs = json_config["epochs"]
    learning_rates = json_config["learning_rates"]
    hipe_datasets = json_config["hipe_datasets"] # Do not iterate over them
    cuda = json_config["cuda"]

    flair.device = f'cuda:{cuda}'

    for seed in seeds:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for learning_rate in learning_rates:
                    run_experiment(seed, batch_size, epoch, learning_rate, hipe_datasets, json_config)  # pylint: disable=no-value-for-parameter
