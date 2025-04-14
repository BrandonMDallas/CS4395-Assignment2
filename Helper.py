import json
import numpy as np

for split in ["training.json", "validation.json", "test.json"]:
    data = json.load(open(split))
    lengths = [ len(item["text"].split()) for item in data ]
    print(f"{split}: avg tokens = {np.mean(lengths):.1f}")