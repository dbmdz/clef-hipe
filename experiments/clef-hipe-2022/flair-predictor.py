import sys
from flair.models import SequenceTagger
from flair.data import Sentence


def get_tags_and_tokens(sent, tagger):
    tokens, tags = [], []
    tagger.predict(sent, label_name="predicted")

    pred_spans = sent.get_spans("predicted")

    for token in sent.tokens:
        pred_tag = "O"
        for span in pred_spans:
            if token in span:
                pred_tag = "B-" + span.tag if token == span[0] else "I-" + span.tag
        tokens.append(token.text)
        tags.append(pred_tag)

    return tokens, tags


test_file, pred_file, ner_model_name = sys.argv[1:]

# load the trained models
ner_model = SequenceTagger.load("dbmdz/" + ner_model_name)

with open(test_file, "rt") as f_p:
    lines = [line.rstrip() for line in f_p]


sentence = []
last_tag = []
sentences = []
hash = []

for index, line in enumerate(lines[2:]):
    if line.startswith("#"):
        if line.startswith('# hipe2022:original_source'):
            hash.append('\n' + line)
        else:
            hash.append(line)

    elif line == '':
        continue

    else:
        if "EndOfSentence" in line:
            sentence.append(line.split('\t')[0])
            last_tag.append(line.split('\t')[-1])

            # coarse ner ner_model
            sentence_flair = Sentence(' '.join(sentence), use_tokenizer=False)
            tokens, tags = get_tags_and_tokens(sentence_flair, ner_model)

            for i in range(len(tokens)):
                sentences.append(tokens[i] + '\t' + tags[i] + '\t_\t_\t_\t_\t_\t_\t' + last_tag[i])

            sentence = []
            last_tag = []

        else:
            sentence.append(line.split('\t')[0])
            last_tag.append(line.split('\t')[-1])

sent_count = 0
hash_count = 0


with open(pred_file, 'w') as p:
    for index, line in enumerate(lines):
        if index == 0 or index == 1:
            p.write(line+'\n')

        elif line == '':
            continue

        elif line.startswith("#"):
            p.write(hash[hash_count]+ '\n')
            hash_count += 1

        else:
            p.write(sentences[sent_count] + '\n')
            sent_count += 1