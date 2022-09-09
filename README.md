# CLEF-HIPE

This repository contains code and models to solve the [HIPE](https://impresso.github.io/CLEF-HIPE-2020/)
(Identifying Historical People, Places and other Entities) evaluation compaign from the [Impresso Project](https://impresso-project.ch/).

It also includes code and models from the following papers:

* [Triple E - Effective Ensembling of Embeddings and Language Models for NER of Historical German](http://ceur-ws.org/Vol-2696/paper_173.pdf)

We are heavily working on better models for historic texts, so please star or watch this repository!

# Triple E - Effective Ensembling of Embeddings and Language Models for NER of Historical German

In this section we give a brief overview of how to reproduce the results from our paper.

As we heavily use Flair and Transformers for our experiments, you should find the relevant scripts in the
`experiments/clef-hipe-2020` folder:

* `word-embeddings`: includes scripts for the experiments with different word embeddings
* `flair-embeddings`: includes scripts for the experiments with different Flair embeddings
* `stacked`: combines word, Flair and Transformer-based embeddings

# hmBERT: Historical Multilingual Language Models for Named Entity Recognition

For [CLEF-HIPE 2022](https://hipe-eval.github.io/HIPE-2022/tasks) we have released multilingual language models: hmBERT.
A detailed description of our final system can be found [here](experiments/clef-hipe-2022/README.md). Additional information
about our released language models for the historic domain is also available [here](hlms.md).

All experiments, released language models and fine-tuned NER models are described in our new
["hmBERT: Historical Multilingual Language Models for Named Entity Recognition"](https://arxiv.org/abs/2205.15575) paper.

# Changelog

* 09.09.2022: Presentation slides of our HIPE-2022 submission can be found [here](https://github.com/dbmdz/clef-hipe/raw/main/experiments/clef-hipe-2022/Pr%C3%A4sentationHISTeria.pdf).
* 03.06.2022: Preprint of our HIPE-2022 system overview paper is available:
              ["hmBERT: Historical Multilingual Language Models for Named Entity Recognition"](https://arxiv.org/abs/2205.15575)
* 22.03.2022: Initial version for our HIPE-2022 submission [here](experiments/clef-hipe-2022/README.md).
* 06.12.2021: Release of smaller multilingual Historic Language Models (ranging from 2-8 layers) - more information [here](hlms.md).
* 18.11.2021: Release of first multilingual and monolingual Historic Language Models - more information [here](hlms.md).
* 04.11.2021: We will take part in the upcoming [CLEF-HIPE 2022](https://hipe-eval.github.io/HIPE-2022/tasks) Shared Task.
              We plan to release new language models before the start of the official shared task very soon.
* 30.10.2021: Manually sentence-segmented Development and Test data for English was added.
* 30.11.2020: Initial version of this repository.

# Citation

You can use the following BibTeX entry for our HIPE-2020 submission:

```bibtex
@inproceedings{DBLP:conf/clef/SchweterM20,
  author    = {Stefan Schweter and
               Luisa M{\"{a}}rz},
  editor    = {Linda Cappellato and
               Carsten Eickhoff and
               Nicola Ferro and
               Aur{\'{e}}lie N{\'{e}}v{\'{e}}ol},
  title     = {Triple {E} - Effective Ensembling of Embeddings and Language Models
               for {NER} of Historical German},
  booktitle = {Working Notes of {CLEF} 2020 - Conference and Labs of the Evaluation
               Forum, Thessaloniki, Greece, September 22-25, 2020},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2696},
  publisher = {CEUR-WS.org},
  year      = {2020},
  url       = {http://ceur-ws.org/Vol-2696/paper\_173.pdf},
  timestamp = {Tue, 27 Oct 2020 17:12:48 +0100},
  biburl    = {https://dblp.org/rec/conf/clef/SchweterM20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

And the following BiBTeX entry for our latest HIPE-2022 submission:

```bibtex
@inproceedings{DBLP:conf/clef/SchweterMSC22,
  author    = {Stefan Schweter and
               Luisa M{\"{a}}rz and
               Katharina Schmid and
               Erion {\c{C}}ano},
  editor    = {Guglielmo Faggioli and
               Nicola Ferro and
               Allan Hanbury and
               Martin Potthast},
  title     = {hmBERT: Historical Multilingual Language Models for Named Entity Recognition},
  booktitle = {Proceedings of the Working Notes of {CLEF} 2022 - Conference and Labs
               of the Evaluation Forum, Bologna, Italy, September 5th - to - 8th,
               2022},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {3180},
  pages     = {1109--1129},
  publisher = {CEUR-WS.org},
  year      = {2022},
  url       = {http://ceur-ws.org/Vol-3180/paper-87.pdf},
  timestamp = {Wed, 10 Aug 2022 16:26:45 +0200},
  biburl    = {https://dblp.org/rec/conf/clef/SchweterMSC22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# Acknowledgements

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC).
Many Thanks for providing access to the TPUs ❤️
