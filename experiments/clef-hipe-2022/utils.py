from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

from pathlib import Path

from typing import List


def prepare_clef_2020_corpus(
    file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
):
    with open(file_in, "rt") as f_p:
        original_lines = f_p.readlines()

    lines = []

    # Add missing newline after header
    lines.append(original_lines[0])

    for line in original_lines[1:]:
        if line.startswith(" \t"):
            # Workaround for empty tokens
            continue

        line = line.strip()

        # Add "real" document marker
        if add_document_separator and line.startswith(document_separator):
            lines.append("-DOCSTART- O")
            lines.append("")

        lines.append(line)

        if eos_marker in line:
            lines.append("")

    # Now here comes the de-hyphenation part ;)
    word_seperator = "¬"
    
    for index, line in enumerate(lines):
        if line.startswith("#"):
            continue

        if line.startswith(word_seperator):
            continue

        if not line:
            continue

        prev_line = lines[index - 1]
        
        prev_prev_line = lines[index - 2]
        
        if not prev_line.startswith(word_seperator):
            continue
    
        # Example:
        # Po  <- prev_prev_line
        # ¬   <- prev_line
        # len <- current_line
        #
        # will be de-hyphenated to:
        # 
        # Polen Dehyphenated-3
        # # ¬
        # # len
        suffix = line.split("\t")[0]

        prev_prev_line_splitted = lines[index - 2].split("\t")
        prev_prev_line_splitted[0] += suffix

        prev_line_splitted = lines[index - 1].split("\t")
        prev_line_splitted[0] = "#" + prev_line_splitted[0]
        prev_line_splitted[-1] += "|Commented"

        current_line_splitted = line.split("\t")
        current_line_splitted[0] = "#" + current_line_splitted[0]
        current_line_splitted[-1] += "|Commented"

        # Add some meta information about suffix length
        # Later, it is possible to re-construct original token and suffix
        prev_prev_line_splitted[9] += f"|Dehyphenated-{len(suffix)}"

        lines[index - 2] = "\t".join(prev_prev_line_splitted)
        lines[index - 1] = "\t".join(prev_line_splitted)
        lines[index]     = "\t".join(current_line_splitted)
    
    # Post-Processing I
    for index, line in enumerate(lines):
        if not line:
            continue

        if not line.startswith(word_seperator):
            continue

        # oh noooo
        current_line_splitted = line.split("\t")
        current_line_splitted[0] = "#" + current_line_splitted[0]

        current_line_splitted[-1] += "|Commented"

        lines[index] = "\t".join(current_line_splitted)
    
    # Post-Processing II
    # Beautify: _|Commented –> Commented
    for index, line in enumerate(lines):
        if not line:
            continue
        
        if not line.startswith("#"):
            continue
        
        current_line_splitted = line.split("\t")
    
        if current_line_splitted[-1] == "_|Commented":
            current_line_splitted[-1] = "Commented"
            lines[index] = "\t".join(current_line_splitted)
    
    # Finally, save it!
    with open(file_out, "wt") as f_out:
        for line in lines:
            f_out.write(line + "\n")

def prepare_newseye_corpus(
    file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
):
    with open(file_in, "rt") as f_p:
        original_lines = f_p.readlines()

    lines = []

    # Add missing newline after header
    lines.append(original_lines[0])

    for line in original_lines[1:]:
        if line.startswith(" \t"):
            # Workaround for empty tokens
            continue

        line = line.strip()

        # Add "real" document marker
        if add_document_separator and line.startswith(document_separator):
            lines.append("-DOCSTART- O")
            lines.append("")

        lines.append(line)

        if eos_marker in line:
            lines.append("")

    # Now here comes the de-hyphenation part ;)
    word_seperator = "¬"
    
    for index, line in enumerate(lines):
        if line.startswith("#"):
            continue

        if not line:
            continue

        last_line = lines[index - 1]
        last_line_splitted = last_line.split("\t")

        if not last_line_splitted[0].endswith(word_seperator):
            continue
    
        # The following example
        #
        # den	O	O	O	null	null	SpaceAfter
        # Ver¬	B-LOC	O	O	null	n                  <- last_line
        # einigten	I-LOC	O	O	null	n	SpaceAfter <- current_line
        # Staaten	I-LOC	O	O	null	n
        # .	O	O	O	null	null
        #
        # will be transformed to:
        #
        # den	O	O	O	null	null	SpaceAfter
        # Vereinigten	B-LOC	O	O	null	n	|Normalized-8
        # #einigten	I-LOC	O	O	null	n	SpaceAfter|Commented
        # Staaten	I-LOC	O	O	null	n
        # .	O	O	O	null	null
    
        suffix = last_line.split("\t")[0].replace(word_seperator, "")  # Will be "Ver" 

        prefix_length = len(line.split("\t")[0])

        # Override last_line:
        # Ver¬ will be transformed to Vereinigten with normalized information at the end

        last_line_splitted[0] = suffix + line.split("\t")[0]

        last_line_splitted[9] += f"|Dehyphenated-{prefix_length}"

        current_line_splitted = line.split("\t")
        current_line_splitted[0] = "#" + current_line_splitted[0]
        current_line_splitted[-1] += "|Commented"

        lines[index - 1] = "\t".join(last_line_splitted)
        lines[index]     = "\t".join(current_line_splitted)

    # Post-Processing I
    # Beautify: _|Commented –> Commented
    for index, line in enumerate(lines):
        if not line:
            continue
        
        if not line.startswith("#"):
            continue
        
        current_line_splitted = line.split("\t")
    
        if current_line_splitted[-1] == "_|Commented":
            current_line_splitted[-1] = "Commented"
            lines[index] = "\t".join(current_line_splitted)
        
    # Finally, save it!
    with open(file_out, "wt") as f_out:
        for line in lines:
            f_out.write(line + "\n")

def prepare_newseye_fi_sv_corpus(
    file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
):
    with open(file_in, "rt") as f_p:
        original_lines = f_p.readlines()

    lines = []

    # Add missing newline after header
    lines.append(original_lines[0])

    for line in original_lines[1:]:
        if line.startswith(" \t"):
            # Workaround for empty tokens
            continue

        line = line.strip()

        # Add "real" document marker
        if add_document_separator and line.startswith(document_separator):
            lines.append("-DOCSTART- O")
            lines.append("")

        lines.append(line)

        if eos_marker in line:
            lines.append("")

    # Now here comes the de-hyphenation part
    # And we want to avoid matching "-DOCSTART-" lines here, so append a tab
    word_seperator = "-\t"

    for index, line in enumerate(lines):
        if line.startswith("#"):
            continue

        if line.startswith(word_seperator):
            continue

        if not line:
            continue

        prev_line = lines[index - 1]

        prev_prev_line = lines[index - 2]

        if not prev_line.startswith(word_seperator):
            continue

        # Example:
        # Po  NoSpaceAfter <- prev_prev_line
        # -   <- prev_line
        # len <- current_line
        #
        # will be de-hyphenated to:
        #
        # Polen Dehyphenated-3
        # # -
        # # len
        #
        # It is really important, that "NoSpaceAfter" in the previous
        # line before hyphenation character! Otherwise, it is no real
        # hyphenation!

        if not "NoSpaceAfter" in prev_line:
            continue

        if not prev_prev_line:
            continue

        suffix = line.split("\t")[0]

        prev_prev_line_splitted = lines[index - 2].split("\t")
        prev_prev_line_splitted[0] += suffix

        prev_line_splitted = lines[index - 1].split("\t")
        prev_line_splitted[0] = "# " + prev_line_splitted[0]
        prev_line_splitted[-1] += "|Commented"

        current_line_splitted = line.split("\t")
        current_line_splitted[0] = "# " + current_line_splitted[0]
        current_line_splitted[-1] += "|Commented"

        # Add some meta information about suffix length
        # Later, it is possible to re-construct original token and suffix
        prev_prev_line_splitted[9] += f"|Dehyphenated-{len(suffix)}"

        lines[index - 2] = "\t".join(prev_prev_line_splitted)
        lines[index - 1] = "\t".join(prev_line_splitted)
        lines[index]     = "\t".join(current_line_splitted)

    # Post-Processing I
    for index, line in enumerate(lines):
        if not line:
            continue

        if not line.startswith(word_seperator):
            continue

        # oh noooo
        current_line_splitted = line.split("\t")
        current_line_splitted[0] = "# " + current_line_splitted[0]

        current_line_splitted[-1] += "|Commented"

        lines[index] = "\t".join(current_line_splitted)

    # Post-Processing II
    # Beautify: _|Commented –> Commented
    for index, line in enumerate(lines):
        if not line:
            continue

        if not line.startswith("#"):
            continue

        current_line_splitted = line.split("\t")

        if current_line_splitted[-1] == "_|Commented":
            current_line_splitted[-1] = "Commented"
            lines[index] = "\t".join(current_line_splitted)

    # Finally, save it!
    with open(file_out, "wt") as f_out:
        for line in lines:
            f_out.write(line + "\n")

def prepare_sonar_corpus(
    file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
):
    with open(file_in, "rt") as f_p:
        lines = f_p.readlines()

    original_dash_symbol = "⸗"
    replaced_dash_symbol = "-"

    with open(file_out, "wt") as f_out:
        # Add missing newline after header
        f_out.write(lines[0] + "\n")

        for line in lines[1:]:
            if line.startswith(" \t"):
                # Workaround for empty tokens
                continue

            line = line.strip()

            # Add "real" document marker
            if add_document_separator and line.startswith(document_separator):
                f_out.write("-DOCSTART- O\n\n")

            if original_dash_symbol in line:
                line = line.replace(original_dash_symbol, replaced_dash_symbol)
                line += "|Normalized"

            f_out.write(line + "\n")

            if eos_marker in line:
                f_out.write("\n")


def prepare_ajmc_corpus(
    file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
):
    with open(file_in, "rt") as f_p:
        lines = f_p.readlines()

    with open(file_out, "wt") as f_out:
        # Add missing newline after header
        f_out.write(lines[0] + "\n")

        for line in lines[1:]:
            if line.startswith(" \t"):
                # Workaround for empty tokens
                continue

            line = line.strip()

            # HIPE-2022 late pre-submission fix:
            # Our hmBERT model has never seen Fraktur, so we replace long s
            line = line.replace("ſ", "s")

            # Add "real" document marker
            if add_document_separator and line.startswith(document_separator):
                f_out.write("-DOCSTART- O\n\n")

            f_out.write(line + "\n")

            if eos_marker in line:
                    f_out.write("\n")

    print("Special preprocessing for AJMC has finished!")


class KBTransformerEmbeddings(TransformerWordEmbeddings):
    def _expand_sentence_with_context(self, sentence):
        if sentence.kb_context == ['']:
            sentence.kb_context = " "
        context_sentence = Sentence(sentence.kb_context[:self.context_length])

        context_length = len(context_sentence)
        expanded_sentence = Sentence([t.text for t in context_sentence.tokens + sentence.tokens], use_tokenizer=False)

        return expanded_sentence, context_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        expanded_sentences = []
        context_offsets = []

        if self.context_length > 0:
            for sentence in sentences:
                # create expanded sentence and remember context offsets
                expanded_sentence, context_offset = self._expand_sentence_with_context(sentence)
                expanded_sentences.append(expanded_sentence)
                context_offsets.append(context_offset)
        else:
            expanded_sentences.extend(sentences)

        self._add_embeddings_to_sentences(expanded_sentences)

        # move embeddings from context back to original sentence (if using context)
        if self.context_length > 0:
            for original_sentence, expanded_sentence, context_offset in zip(
                sentences, expanded_sentences, context_offsets
            ):
                if self.token_embedding:
                    for token_idx, token in enumerate(original_sentence):
                        token.set_embedding(
                            self.name,
                            expanded_sentence[token_idx + context_offset].get_embedding(self.name),
                        )
                if self.document_embedding:
                    original_sentence.set_embedding(self.name, expanded_sentence.get_embedding(self.name))


if __name__ == "__main__":
    # Please make sure, that you've cleaned the dataset folder when using own preprocessing function
    from flair.datasets import NER_HIPE_2022

    # HIPE-2020
    hipe_2020_de = NER_HIPE_2022(dataset_name="hipe2020",
                                 language="de",
                                 version="v2.0",
                                 add_document_separator=True,
                                 preproc_fn=prepare_clef_2020_corpus)
    hipe_2020_en = NER_HIPE_2022(dataset_name="hipe2020",
                                 language="en",
                                 version="v2.0",
                                 add_document_separator=True,
                                 preproc_fn=prepare_clef_2020_corpus)
    hipe_2020_fr = NER_HIPE_2022(dataset_name="hipe2020",
                                 language="fr",
                                 version="v2.0",
                                 add_document_separator=True,
                                 preproc_fn=prepare_clef_2020_corpus)

    # NewsEye - German and French
    newseye_de = NER_HIPE_2022(dataset_name="newseye",
                               language="de",
                               version="v2.0",
                               add_document_separator=True,
                               preproc_fn=prepare_newseye_corpus)
    newseye_fr = NER_HIPE_2022(dataset_name="newseye",
                               language="fr",
                               version="v2.0",
                               add_document_separator=True,
                               preproc_fn=prepare_newseye_corpus)

    # NewsEye - Swedish and Finnish uses another hyphenation format
    newseye_fi = NER_HIPE_2022(dataset_name="newseye",
                               language="fi",
                               version="v2.0",
                               add_document_separator=True,
                               preproc_fn=prepare_newseye_fi_sv_corpus)
    newseye_sv = NER_HIPE_2022(dataset_name="newseye",
                               language="sv",
                               version="v2.0",
                               add_document_separator=True,
                               preproc_fn=prepare_newseye_fi_sv_corpus)

    # Sonar only gets some basic normalization
    sonar_de = NER_HIPE_2022(dataset_name="sonar",
                            language="de",
                            version="v2.0",
                            add_document_separator=True,
                            preproc_fn=prepare_sonar_corpus)

    # Number of sentences and documents should never change when doing de-hyphenation!
    # Check that here:
    assert len(hipe_2020_de.train) == 3470 + 2 + 103
    assert len(hipe_2020_de.dev) == 1202 + 33
    assert len(hipe_2020_en.dev) == 1045 + 80
    assert len(hipe_2020_fr.train) == 5743 + 158
    assert len(hipe_2020_fr.dev) == 1244 + 43

    assert len(newseye_de.train) == 20839 + 1 + 7
    assert len(newseye_de.dev) == 1110 + 1 + 12
    assert len(newseye_fr.train) == 7106 + 1 + 35
    assert len(newseye_fr.dev) == 662 + 1 + 35

    assert len(newseye_fi.train) == 1141 + 1 + 24
    assert len(newseye_fi.dev) == 140 + 1 + 24
    assert len(newseye_sv.train) == 1063 + 1 + 21
    assert len(newseye_sv.dev) == 126 + 1 + 21

    assert len(sonar_de.dev) == 816 + 10 + 10
