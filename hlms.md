# Historic Language Models (HLMs)

Our Historic Language Models Zoo contains support for the following languages - incl. their training data source:

| Language | Training data | Size 
| -------- | ------------- | ----
| German   | [Europeana](http://www.europeana-newspapers.eu/)       | 13-28GB (filtered)
| French   | [Europeana](http://www.europeana-newspapers.eu/)       | 11-31GB (filtered)
| English  | [British Library](https://data.bl.uk/digbks/db14.html) | 24GB (year filtered)
| Finnish  | [Europeana](http://www.europeana-newspapers.eu/)       | 1.2GB
| Swedish  | [Europeana](http://www.europeana-newspapers.eu/)       | 1.1GB

# Corpora Stats

## German Europeana Corpus

We provide some statistics using different thresholds of ocr confidences, in order to shrink down the corpus size
and use less-noisier data:

| OCR confidence | Size
| -------------- | ----
| **0.60**       | 28GB
| 0.65           | 18GB
| 0.70           | 13GB

For the final corpus we use a OCR confidence of 0.6 (28GB). The following plot shows a tokens per year distribution:

![German Europeana Corpus Stats](stats/figures/german_europeana_corpus_stats.png)

## French Europeana Corpus

Like German, we use different ocr confidence thresholds:

| OCR confidence | Size
| -------------- | ----
| 0.60           | 31GB
| 0.65           | 27GB
| **0.70**       | 27GB
| 0.75           | 23GB
| 0.80           | 11GB

For the final corpus we use a OCR confidence of 0.7 (27GB). The following plot shows a tokens per year distribution:

![French Europeana Corpus Stats](stats/figures/french_europeana_corpus_stats.png)

## British Library Corpus

Metadata is taken from [here](https://data.bl.uk/digbks/DB21.html). Stats incl. year filtering:

| Years             | Size
| ----------------- | ----
| ALL               | 24GB
| >= 1800 && < 1900 | 24GB

We use the year filtered variant. The following plot shows a tokens per year distribution:

![British Library Corpus Stats](stats/figures/bl_corpus_stats.png)

## Finnish Europeana Corpus

| OCR confidence | Size
| -------------- | ----
| 0.60           | 1.2GB

The following plot shows a tokens per year distribution:

![Finnish Europeana Corpus Stats](stats/figures/finnish_europeana_corpus_stats.png)

## Swedish Europeana Corpus

| OCR confidence | Size
| -------------- | ----
| 0.60           | 1.1GB

The following plot shows a tokens per year distribution:

![Swedish Europeana Corpus Stats](stats/figures/swedish_europeana_corpus_stats.png)

## All Corpora

The following plot shows a tokens per year distribution of the complete training corpus:

![All Corpora Stats](stats/figures/all_corpus_stats.png)

# Multilingual Vocab generation

For the first attempt, we use the first 10GB of each pretraining corpus. We upsample both Finnish and Swedish to ~10GB.
The following tables shows the exact size that is used for generating a 32k and 64k subword vocabs:

| Language | Size
| -------- | ----
| German   | 10GB
| French   | 10GB
| English  | 10GB
| Finnish  | 9.5GB
| Swedish  | 9.7GB

We then calculate the subword fertility rate and portion of `[UNK]`s over the following NER corpora:

| Language | NER corpora
| -------- | ------------------
| German   | CLEF-HIPE, NewsEye
| French   | CLEF-HIPE, NewsEye
| English  | CLEF-HIPE
| Finnish  | NewsEye
| Swedish  | NewsEye

Breakdown of subword fertility rate and unknown portion per language for the 32k vocab:

| Language | Subword fertility  | Unknown portion
| -------- | ------------------ | ---------------
| German   | 1.43               | 0.0004
| French   | 1.25               | 0.0001
| English  | 1.25               | 0.0
| Finnish  | 1.69               | 0.0007
| Swedish  | 1.43               | 0.0

Breakdown of subword fertility rate and unknown portion per language for the 64k vocab:

| Language | Subword fertility  | Unknown portion
| -------- | ------------------ | ---------------
| German   | 1.31               | 0.0004
| French   | 1.16               | 0.0001
| English  | 1.17               | 0.0
| Finnish  | 1.54               | 0.0007
| Swedish  | 1.32               | 0.0

# Final pretraining corpora

We upsample Swedish and Finnish to ~27GB. The final stats for all pretraining corpora can be seen here:

| Language | Size
| -------- | ----
| German   | 28GB
| French   | 27GB
| English  | 24GB
| Finnish  | 27GB
| Swedish  | 27GB

Total size is 130GB.

# Pretraining

We train a multilingual BERT model using the 32k vocab with the official BERT implementation
on a v3-32 TPU using the following parameters:

```bash
python3 run_pretraining.py --input_file gs://histolectra/historic-multilingual-tfrecords/*.tfrecord \
--output_dir gs://histolectra/bert-base-historic-multilingual-cased \
--bert_config_file ./config.json \
--max_seq_length=512 \
--max_predictions_per_seq=75 \
--do_train=True \
--train_batch_size=128 \
--num_train_steps=3000000 \
--learning_rate=1e-4 \
--save_checkpoints_steps=100000 \
--keep_checkpoint_max=20 \
--use_tpu=True \
--tpu_name=electra-2 \
--num_tpu_cores=32
```

The following plot shows the pretraining loss curve:

![Training loss curve](stats/figures/pretraining_loss.png)
