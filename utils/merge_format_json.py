import json
import os



# 合并所有json文件为一个jsonl文件
# res=[]
# for item in os.listdir('data/KodCode_splited/ByStyle'):
#     with open(f'data/KodCode_splited/ByStyle/{item}', 'r') as f:
#         data = json.load(f)
#         for item in data:
#             res.append(item)
# with open('data/KodCode_all.jsonl', 'w') as f:
#     for item in res:
#         f.write(json.dumps(item) + '\n')
    

# 加载jsonl文件并修改所有对象的键名
i=0
with open('data/KodCode_all.jsonl', 'r') as f:
    for line in f:
        i+=1
        item = json.loads(line)
        item['prompt'] = item['question']
        item['target'] = item['solution']
        del item['question']
        del item['solution']
        with open('data/KodCode_all_formatted.jsonl', 'a') as f:
            f.write(json.dumps(item) + '\n')
        
print(i)    