import sys
import random

from typing import List

filename = sys.argv[1] # like: data/future/dev-v1.2/en/HIPE-data-v1.2-dev-en-normalized-manual-eos.tsv
language = sys.argv[2] # like: en
splits = int(sys.argv[3]) # like 5

def save_document(filename: str, sentences: List[str]) -> None:
    with open(filename, "wt") as f_p:
        # Header:
        header = "TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC"
        f_p.write(header + "\n\n")
        for sentence in sentences:
            f_p.writelines("\n".join(sentence))
            f_p.write("\n\n")

for seed in range(splits):
    all_documents = []
    current_document = []

    with open(filename, "rt") as f_p:
        for line in f_p:
            line = line.rstrip()

            if line.startswith("TOKEN"):
                continue

            if line.startswith(f"# language = {language}"):
                # New document started

                if current_document:
                    all_documents.append(current_document)
                    current_document = []
                
                current_document.append(line)
                continue

            current_document.append(line)
        
        if current_document:
            all_documents.append(current_document)
            current_document = []

    random.seed(seed)
    random.shuffle(all_documents)

    length = len(all_documents)

    shard_size = length // 10

    train_boundary = shard_size * 8
    dev_boundary = train_boundary + shard_size

    new_train_sentences = all_documents[:train_boundary]
    new_dev_sentences = all_documents[train_boundary:]

    current_train_file = f"./data/future/training-v1.2/{language}/train_random_split_{seed}.tsv"
    current_dev_file = f"./data/future/dev-v1.2/{language}/dev_random_split_{seed}.tsv"

    print(f"Writing out files for seed {seed}...")
    save_document(current_train_file, new_train_sentences)
    save_document(current_dev_file, new_dev_sentences)