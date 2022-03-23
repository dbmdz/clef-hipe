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
