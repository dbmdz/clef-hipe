import sys

from nltk.tokenize import sent_tokenize


filename, language, min_tokens_per_sentence = sys.argv[1:]

min_tokens_per_sentence = int(min_tokens_per_sentence)

with open(filename, "rt") as f_p:
    for line in f_p:
        line = line.rstrip()

        if not line:
            continue

        for sentence in sent_tokenize(line, language):
            if len(sentence.split()) >= min_tokens_per_sentence:
                print(sentence)
