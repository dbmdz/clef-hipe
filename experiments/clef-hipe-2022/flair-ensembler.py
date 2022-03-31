import sys

from collections import Counter
from typing import List

system_outputs = sys.argv[1:]

def collector(filename: str, column_id: int = 2) -> List[str]:
    lines = []
    
    with open(filename, "rt") as f_p:
        for line in f_p:
            line = line.strip()
            
            if not line or line.startswith("#"):
                lines.append(line)
                continue
            
            line_splitted = line.split(" ")
            
            lines.append(line_splitted[column_id])
    return lines

token_list = collector(filename=system_outputs[0], column_id=0)
gold_label_list = collector(filename=system_outputs[0], column_id=1)

system_predictions = [ collector(filename=filename, column_id=2) for filename in system_outputs ]

for index, _ in enumerate(token_list):
    token = token_list[index]
    gold_label = gold_label_list[index]

    if not token or token.startswith("#"):
        print("")
        continue

    label_counter = Counter()

    for system_prediction in system_predictions:
        predicted_label = system_prediction[index]

        label_counter[predicted_label] += 1
    
    best_label = label_counter.most_common(1)[0][0]

    print(token, gold_label, best_label)
