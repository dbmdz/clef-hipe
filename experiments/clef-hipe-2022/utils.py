from pathlib import Path

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

if __name__ == "__main__":
    # Please make sure, that you've cleaned the dataset folder when using own preprocessing function
    from flair.datasets import NER_HIPE_2022

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

    # Number of sentences and documents should never change when doing de-hyphenation!
    # Check that here:
    assert len(hipe_2020_de.train) == 3470 + 2 + 103
    assert len(hipe_2020_de.dev) == 1202 + 33
    assert len(hipe_2020_en.dev) == 1045 + 80
    assert len(hipe_2020_fr.train) == 5743 + 158
    assert len(hipe_2020_fr.dev) == 1244 + 43
