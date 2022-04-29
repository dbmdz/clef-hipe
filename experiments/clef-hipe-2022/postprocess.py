import string

import click


@click.command()
@click.argument("predictions", type=click.File("r"))
@click.argument("processed_predictions", type=click.File("w"))
def postprocess(predictions, processed_predictions):
    # We assume that names of these entity types don't start with a punctuation
    # mark and remove B-<type_to_correct> annotations from punctuation marks
    types_to_correct = ["pers", "loc", "date", "work"]
    corrected_type = ""
    for line in predictions:
        line = line.strip()
        # Headers and comments
        if line.startswith("TOKEN") or line.startswith("#"):
            processed_predictions.write(line + "\n")
            corrected_type = ""
            continue
        # Empty lines
        if not line:
            processed_predictions.write("\n")
            corrected_type = ""
            continue
        elems = [e.strip() for e in line.split("\t")]
        # Remove B-<type> annotations from punctuation marks
        if elems[0] in string.punctuation:
            if elems[1].startswith("B-") and elems[1][2:] in types_to_correct:
                corrected_type = elems[1][2:]
                elems[1] = "O"
                processed_predictions.write("\t".join(elems) + "\n")
                continue
        # Replace I-<type> with B-<type> annotations following corrections
        if (
            corrected_type
            and elems[1].startswith("I-")
            and elems[1][2:] == corrected_type
        ):
            elems[1] = f"B-{corrected_type}"
        processed_predictions.write("\t".join(elems) + "\n")
        corrected_type = ""


if __name__ == "__main__":
    postprocess()
