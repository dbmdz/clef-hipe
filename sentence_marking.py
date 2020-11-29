import sys

filename = sys.argv[1]

potential_eos = list(".:;?!")

with open(filename, "rt") as f_p:
    lines = [line.rstrip() for line in f_p]

found_eos_positions = set()

for index, line in enumerate(lines):
    if not line:
        continue

    line_splitted = line.split("\t")

    if len(line_splitted) < 2:
        continue

    current_token = line_splitted[0]
    current_label = line_splitted[1]

    if current_token in potential_eos and index + 1 < len(lines):
        # Oh boy, do not break entities!!!!
        if current_label not in ["O", "_"]:
            continue

        prev_token = lines[index - 1].split("\t")[0]
        next_token = lines[index + 1].split("\t")[0]

        # Manual ruleset

        # prev_token has only one char, no EOS
        if len(prev_token) == 1:
            continue

        # prev_token only consists of digits, no EOS
        if prev_token.isdigit():
            continue

        # Do we need a rule for EOS == only when next_token[0] is upper 🤔

        if len(line_splitted) == 10:
            found_eos_positions.add(index)

for index, line in enumerate(lines):
    eos_found = False
    if index in found_eos_positions:
        line += "|EOS"
        eos_found = True
    
    print(line)

    if eos_found:
        print("")

