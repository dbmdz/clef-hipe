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
$ cd flair && git checkout 88dc52f && pip3 install -e . && cd ..

# Clone this repository
$ git clone https://github.com/dbmdz/clef-hipe.git
$ cd clef-hipe/experiments/clef-hipe-2022

# Run training!
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

Configuration: 1166 train + 165 dev sentences (incl. doc marker).
Label set: `[HumanProd, LOC, ORG, PER]`.

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs4-e10-lr5e-05`  | hmBERT         | 75.34

### Swedish

Configuration: 1085 train + 148 dev sentcnes (incl. doc marker).
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

Configuration: 14465 train + 1392 dev sentences (incl. doc marker).
Label set: `[loc, org, pers]`.

| Best configuration | Language Model | Result
| ------------------ | -------------- | ------
| `bs8-e10-lr5e-05`  | hmBERT         | 65.53

### Ensembling

| Single Model | Ensemble (Single Model)
| ------------ | -----------------------
| 65.53        | 66.41

## TopRes19th

Configuration: 6183 train + 680 dev sentences (incl. doc marker).
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
