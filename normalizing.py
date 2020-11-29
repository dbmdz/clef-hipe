import sys

filename = sys.argv[1]

word_seperator = "¬"

with open(filename, "rt") as f_p:
    lines = [line.rstrip() for line in f_p]

for index, line in enumerate(lines):
    if line.startswith("#"):
        continue

    if line.startswith(word_seperator):
        continue

    if not line:
        continue

    last_line = lines[index - 1]

    if not last_line.startswith("#"):
        continue

    second_last_line = lines[index - 2]

    if not second_last_line.startswith(word_seperator):
        continue

    # Example:
    # Po  <- third_last_line
    # ¬   <- second_last_line
    # #   <- last_line
    # len <- current_line

    # We need to modify third_last_line + append token of current_line as suffix
    # All other previous lines are commented out
    suffix = line.split("\t")[0]

    third_last_line_splitted = lines[index - 3].split("\t")
    third_last_line_splitted[0] += suffix

    second_last_line_splitted = lines[index - 2].split("\t")
    second_last_line_splitted[0] = "#" + second_last_line_splitted[0]
    second_last_line_splitted[-1] += "|Commented"

    last_line_splitted = lines[index - 1].split("\t")
    last_line_splitted[-1] += "|Commented"

    current_line_splitted = line.split("\t")
    current_line_splitted[0] = "#" + current_line_splitted[0]
    current_line_splitted[-1] += "|Commented"

    # Add some meta information about suffix length
    # Later, it is possible to re-construct original token and suffix
    third_last_line_splitted[9] += f"|Normalized-{len(suffix)}"

    lines[index - 3] = "\t".join(third_last_line_splitted)
    lines[index - 2] = "\t".join(second_last_line_splitted)
    lines[index - 1] = "\t".join(last_line_splitted)
    lines[index]     = "\t".join(current_line_splitted)

# oh boy... sanitize
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

print("\n".join(lines))

