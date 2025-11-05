import json
import os
import random

with open('data/KodCode_all_formatted.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

test_data = random.sample(data, 5000)
with open('data/KodCode_train_5000.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')
#然后删除all中的test_data，存到KodCode_train.jsonl
# for item in test_data:
#     data.remove(item)
# with open('data/KodCode_train.jsonl', 'w') as f:
#     for item in data:
#         f.write(json.dumps(item) + '\n')