import re
import sys
import numpy as np

from collections import defaultdict
from pathlib import Path

#pattern = "bert-tiny-historic-multilingual-cased-*"  # sys.argv[1]
pattern = sys.argv[1]

log_dirs = Path("./").rglob(f"{pattern}")

dev_results = defaultdict(list)
test_results = defaultdict(list)

for log_dir in log_dirs:
    training_log = log_dir / "training.log"
    

    if not training_log.exists():
        print(f"No training.log found in {log_dir}")
    
    matches = re.match(".*(bs.*?)-(ws.*?)-(e.*?)-(lr.*?)-layers-1-crfFalse-(\d+)", str(log_dir))
    
    batch_size = matches.group(1)
    ws = matches.group(2)
    epochs = matches.group(3)
    lr = matches.group(4)
    seed = matches.group(5)
    
    result_identifier = f"{ws}-{batch_size}-{epochs}-{lr}"
    
    with open(training_log, "rt") as f_p:
        all_dev_results = []
        for line in f_p:
            line = line.rstrip()

            if "f1-score (micro avg)" in line:
                dev_result = line.split(" ")[-1]
                all_dev_results.append(dev_result)
                #dev_results[result_identifier].append(dev_result)
            
            if "F-score (micro" in line:
                test_result = line.split(" ")[-1]
                test_results[result_identifier].append(test_result)

        best_dev_result = max([float(value) for value in all_dev_results])
        dev_results[result_identifier].append(best_dev_result)
                
mean_dev_results = {}

print("Debug:", dev_results)

for dev_result in dev_results.items():
    result_identifier, results = dev_result
    
    mean_result = np.mean( [float(value) for value in results])
    
    mean_dev_results[result_identifier] = mean_result


print("Averaged Development Results:")

sorted_mean_dev_results = dict(sorted(mean_dev_results.items(), key=lambda item: item[1], reverse=True))

for mean_dev_config, score in sorted_mean_dev_results.items():
    print(f"{mean_dev_config} : {round(score * 100, 2)}")

best_dev_configuration = max(mean_dev_results, key=mean_dev_results.get)

print("")

print("Best configuration:", best_dev_configuration)

print("")

print("Best Development Score:",
      round(mean_dev_results[best_dev_configuration] * 100, 2))

