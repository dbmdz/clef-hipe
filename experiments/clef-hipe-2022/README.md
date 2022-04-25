# HIPE-2022

This readme contains the documentation for our HIPE-2022 submission. Stay tuned!

# Setup

In order to get reproducible results across different machines, we pin the following
dependencies:

| Library/Dependency | Commit/Version
| ------------------ | --------------
| Transformers       | [`12428f0`](https://github.com/huggingface/transformers/commit/12428f0ef15bb3631e7a5f04672ddb05f363de97)
| Flair              | [`88dc52f`](https://github.com/flairNLP/flair/commit/88dc52ff478f627fd87c9bf971bd6e3042631093)
| NVIDIA PyTorch     | `nvcr.io/nvidia/pytorch:22.01-py3`

We recommend using NVIDIA Docker in order to train models. A typical docker setup can be started with:

```bash
$ docker run --gpus all --shm-size 64G -v /home/arthur:/mnt -it --rm nvcr.io/nvidia/pytorch:22.01-py3 /bin/bash
```

Please adjust your desired mountpoints.

After entering the container, just install the following dependencies:

```bash
$ cd /mnt
$ git clone https://github.com/huggingface/transformers.git
$ cd transformers && git checkout 12428f0 && pip3 install -e . && cd ..
$ git clone https://github.com/flairNLP/flair.git
$ cd flair && git checkout 08f14ff && pip3 install -e . && cd ..

# Clone this repository
$ git clone https://github.com/dbmdz/clef-hipe.git
$ cd clef-hipe/experiments/clef-hipe-2022

# Run training!
$ python3 flair-fine-tuner.py ./configs/ajmc/ajmc_hmbert_de.json

# Get best configuration
$ python3 flair-log-parser.py "hipe2022-flert-fine-tune-ajmc/de-dbmdz/bert-base-historic-multilingual-cased-*"
```

# Baseline experiments

We perform a (non-extensive) hyper-parameter search:

| Parameter     | Values
| ------------- | ------
| Batch Size    | `[4, 8]`
| Epoch         | `[5, 10]`
| Learning Rate | `[3e-5, 5e-5]`
| Seed          | `[1, 2, 3, 4, 5]`

For each language, 40 models are trained. The script `flair-log-parser.py` parses all training logs and outputs the best
configuration (Batch Size, Epoch and Learning Rate) averaged over all seeds.

We use version 2 of the HIPE-2022 datasets for baseline experiments.

## AJMC

### German

[Configuration](configs/ajmc/ajmc_hmbert_de.json): 1100 train + 206 dev sentences (incl. doc marker).
Label set: `[scope, pers, work, loc, object, date]`.

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs4-e10-lr5e-05`  | hmBERT         | 86.21

### English

[Configuration](configs/ajmc/ajmc_hmbert_en.json): 1214 train + 266 dev sentences (incl. doc marker).
Label set: `[scope, pers, work, loc, date, object]`.

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs8-e10-lr5e-05`  | hmBERT         | 84.98

### French

[Configuration](configs/ajmc/ajmc_hmbert_fr.json): 966 train + 219 dev sentences (incl. doc marker).
Label set: `[scope, pers, work, loc, object, date]`.

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs8-e10-lr5e-05`  | hmBERT         | 85.69

### One Model

In this experiment, we use the training and development data from all languages (German, English and French)
to train one model. We report best configuration here and perform a detailed per-language analysis later on:

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs4-e10-lr5e-05`  | hmBERT         | 85.69

Then we use all 5 models from the `bs4-e10-lr5e-05` configuration, evaluate them for each language and report
averaged F1-Score:

| Language | Result
| -------- | ------
| German   | 86.68
| English  | 84.85
| French   | 85.09

Comparison table between "one model" vs. "single model" for AJMC:

| Language | Single Model | One Model
| -------- | ------------ | ---------
| German   | 86.21        | 86.68
| English  | 84.98        | 84.85
| French   | 85.69        | 85.09

### Ensembling

The script `flair-ensembler.py` can be used to ensemble made predictions from Flair via simple majority vote,
based on the Flair predictions in `dev.tsv` for each run:

| Language | Single Model | Ensemble (Single Model)
| -------- | ------------ | -----------------------
| German   | 86.21        | 86.85
| English  | 84.98        | 86.39
| French   | 85.69        | 85.86

For the ensembling approach, the official CoNLL-2003 evaluation script is used.

## NewsEye

### Finnish

[Configuration](./configs/newseye/newseye_hmbert_fi.json): 1166 train + 165 dev sentences (incl. doc marker).
Label set: `[HumanProd, LOC, ORG, PER]`.

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs4-e10-lr5e-05`  | hmBERT         | 75.34

### Swedish

[Configuration](./configs/newseye/newseye_hmbert_sv.json): 1085 train + 148 dev sentcnes (incl. doc marker).
Label set: `[HumanProd, LOC, ORG, PER]`.

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs4-e10-lr3e-05`  | hmBERT         | 80.63

### One Model

In this experiment, we use the training and development data from all languages
(Swedish and Finnish) to train one model. We report best configuration here and
perform a detailed per-language analysis later on:

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs8-e10-lr5e-05`  | hmBERT         | 80.15

hen we use all 5 models from the `bs8-e10-lr5e-05` configuration, evaluate them
for each language and report averaged F1-Score:

| Language | Result
| -------- | ------
| Finnish  | 78.51
| Swedish  | 81.53

### Ensembling

| Language | Single Model | Ensemble (Single Model)
| -------- | ------------ | -----------------------
| Finnish  | 75.34        | 76.63
| Swedish  | 80.63        | 79.93

## LeTemps

[Configuration](./configs/letemps/letemps_hmbert_fr.json): 14465 train + 1392 dev sentences (incl. doc marker).
Label set: `[loc, org, pers]`.

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs8-e10-lr5e-05`  | hmBERT         | 65.53

### Ensembling

| Single Model | Ensemble (Single Model)
| ------------ | -----------------------
| 65.53        | 66.41

## TopRes19th

[Configuration](./configs/topres19th/topres19th_hmbert_en.json): 6183 train + 680 dev sentences (incl. doc marker).
Label set: `[BUILDING, LOC, STREET]`.

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs8-e10-lr5e-05`  | hmBERT         | 80.98

### Ensembling

| Single Model | Ensemble (Single Model)
| ------------ | -----------------------
| 80.98        | 81.39

# Baseline experiments - Word Embeddings

We use the official FastText Embeddings from CLEF-HIPE 2020 to train
baseline models, which can be found
[here](https://files.ifi.uzh.ch/cl/siclemat/impresso/clef-hipe-2020/fasttext/).

## AJMC

We use the standard feature-base approach for training a model with
FastText embeddings only.

Notice: we needed to modify the `FastText()` instance due to an
implementation error with latest Gensim version.

Hyper-param search is restricted to different batch sizes (4 and 8).

### German

| Best configuration | Word Embeddings (Size)                           | Result
| ------------------ | ------------------------------------------------ | ------
| `bs4-e200-lr0.1`   | `de-model-skipgram-300-minc20-ws5-maxn-0` (2.4G) | 68.64
| `bs8-e200-lr0.1`   | `de-model-skipgram-300-minc20-ws5-maxn-6` (14G)  | 75.71

### French

| Best configuration | Word Embeddings (Size)                           | Result
| ------------------ | ------------------------------------------------ | ------
| `bs4-e200-lr0.1`   | `fr-model-skipgram-300minc20-ws5-maxn-0` (1.8G)  | 56.88
| `bs4-e200-lr0.1`   | `fr-model-skipgram-300minc20-ws5-maxn-6` (13G)   | 76.58

### English

| Best configuration | Word Embeddings (Size)                           | Result
| ------------------ | ------------------------------------------------ | ------
| `bs4-e200-lr0.1`   | `en-model-skipgram-300-minc0-ws5-maxn-0` (0.9G)  | 63.10
| `bs4-e200-lr0.1`   | `en-model-skipgram-300-minc0-ws5-maxn-6` (2.1G)  | 76.76

# KB-NER experiments

We extract additional contexts for training and development instances using the
[KB-NER](https://github.com/Alibaba-NLP/KB-NER) implementation and fine-tune a
model for all three languages on the AJMC dataset:

| Best configuration | Language Model | KB context size | Result
| ------------------ | -------------- | --------------- | ------
| `bs4-e10-lr3e-05`  | hmBERT         | 128             | 85.30
| `bs4-e10-lr5e-05`  | hmBERT         | 256             | 85.07

The current baseline SOTA for the one-model approach is 85.69. The KB-NER approach
is worse than the baseline model, and also needs more computing resources
(GPU RAM, fine-tuning time). For this reason, we do not use the KB-NER approach for
our final submission.

The script `flair-fine-tuner-kb.py` can be used for fine-tuning models with the KB-NER
approach. The corresponding configuration file is located under
`./configs/ajmc/ajmc_hmbert_all_kb.json`.


Technically, we did overload the `TransformerWordEmbeddings` instance from
Flair to get fine-tuning working with left-contexts (coming from the KB).

# Multistage Fine-Tuning

Inspired by the [KB-NER](https://arxiv.org/abs/2203.00545) paper, we use a multistage
fine-tuning approach for our final submission.

In the first stage, we fine-tune one multi-lingual model over the training and
development of all three languages (German, English and French). Then we select the
best hyper-parameter configuration (combination of batch size, number of epochs and
learning rate). This will result in 5 best models (5 because of the number of random
seeds). Each of these best models are fine-tuned in the second stage:

The second stage will fine-tune models for each language (instead of fine-tuning one
model over all languages). In our preliminary results, this will heavily boost
performance:

| Seed | Stage 1 | Stage 2 - German | Stage 2 - English | Stage 2 - French
| ---- | ------- | ---------------- | ----------------- | ----------------
| 1    | 85.62   | 87.79            | 86.48             | 86.43
| 2    | 85.93   | 86.59            | 86.51             | 87.03
| 3    | 85.34   | 87.39            | 85.65             | 85.67
| 4    | 86.15   | 87.32            | 85.68             | 86.99
| 5    | 86.93   | 87.65            | 87.07             | 87.21
| Avg. | 85.99   | 87.35            | 86.28             | 86.67

Here's a performance comparison between single model, one model and multistage fine-tuning:

| Language | Single Model | One Model | Multistage
| -------- | ------------ | --------- | --------------
| German   | 86.21        | 86.68     | 87.35 (+0.67%)
| English  | 84.98        | 84.85     | 86.28 (+1.43%)
| French   | 85.69        | 85.09     | 86.67 (+1.58%)

Diff of multistage is compared against one model performance. Performance boost of
multistage approach is ~1.23% (average) compared to one model approach. Thus, we use
the multistage fine-tuning approach for our final submission.

We select the best-performing models within the best hyper-parameter configuration and
upload the model to the Hugging Face Model Hub:

| Language | Model Hub Link                                               | Flair identifier          | F1-Score | Configuration
| -------- | ------------------------------------------------------------ | ------------------------- | -------- | -----------------------
| German   | [here](https://huggingface.co/dbmdz/flair-hipe-2022-ajmc-de) | `flair-hipe-2022-ajmc-de` | 88.29    | `bs8-e5-lr3e-05-seed2`
| English  | [here](https://huggingface.co/dbmdz/flair-hipe-2022-ajmc-en) | `flair-hipe-2022-ajmc-en` | 87.29    | `bs8-e10-lr3e-05-seed5`
| French   | [here](https://huggingface.co/dbmdz/flair-hipe-2022-ajmc-fr) | `flair-hipe-2022-ajmc-fr` | 88.06    | `bs8-e10-lr3e-05-seed4`

The model can be loaded with:

```python
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger.load("dbmdz/flair-hipe-2022-ajmc-de")
```

This will automatically download the model from the Hugging Face Model Hub, so it can be used within Flair. Additionally, all fine-tuning
parameters can be displayed with the `.print_model_card()` function:

```python
tagger.print_model_card()
```
