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

from utils import prepare_clef_2020_corpus

def run_experiment(seed, batch_size, epoch, learning_rate, hipe_datasets, json_config):
    # Config values
    # Replace it with more Pythonic solutions later!
    hf_model = json_config["hf_model"]
    context_size = json_config["context_size"]
    layers = json_config["layers"] if "layers" in json_config else "-1"
    use_crf = json_config["use_crf"] if "use_crf" in json_config else False
    additional_hipe_datasets = json_config["additional_hipe_datasets"] if "additional_hipe_datasets" in json_config else None
    label_name_map = json_config["label_name_map"] if "label_name_map" in json_config else None

    # Set seed for reproducibility
    set_seed(seed)

    corpus_list = [] 

    # Dataset-related
    for dataset in hipe_datasets:
        dataset_name, language = dataset.split("/")
        corpus_list.append(NER_HIPE_2022(dataset_name=dataset_name, language=language, add_document_separator=True))

    if additional_hipe_datasets and label_name_map:
        # Special case: do not use Dev data from additional datasets
        # This makes evaluation and comparison much more easier!

        for dataset in additional_hipe_datasets:
            dataset_name, language = dataset.split("/")

            preproc_fn = None

            if dataset_name == "hipe2020":
                print("Using own HIPE-2020 Preprocessing function.")
                print("Please make sure that Flair Datasets folder was cleaned before!")
                preproc_fn = prepare_clef_2020_corpus

            additional_corpus = NER_HIPE_2022(dataset_name=dataset_name, label_name_map=label_name_map,
                                              language=language, add_document_separator=True,
                                              preproc_fn=preproc_fn)
            additional_corpus._dev = []
            corpus_list.append(additional_corpus)

    if context_size == 0:
        context_size = False

    print("FLERT Context:", context_size)
    print("Layers:", layers)
    print("Use CRF:", use_crf)

    corpora: MultiCorpus = MultiCorpus(corpora=corpus_list, sample_missing_splits=False)
    label_dictionary = corpora.make_label_dictionary(label_type="ner")
    print("Label Dictionary:", label_dictionary.get_items())

    # Embeddings
    embeddings = TransformerWordEmbeddings(
        model=hf_model,
        layers=layers,
        subtoken_pooling="first",
        fine_tune=True,
        use_context=context_size,
    )

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dictionary,
        tag_type="ner",
        use_crf=use_crf,
        use_rnn=False,
        reproject_embeddings=False,
    )

    # Trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpora)

    datasets = "-".join([dataset for dataset in hipe_datasets])

    trainer.fine_tune(
        f"hipe2022-flert-fine-tune-{datasets}-{hf_model}-bs{batch_size}-ws{context_size}-e{epoch}-lr{learning_rate}-layers{layers}-crf{use_crf}-{seed}",
        learning_rate=learning_rate,
        mini_batch_size=batch_size,
        max_epochs=epoch,
        shuffle=True,
        embeddings_storage_mode='none',
        weight_decay=0.,
        use_final_model_for_eval=False,
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
