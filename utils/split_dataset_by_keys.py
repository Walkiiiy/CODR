import json
import os
from collections import defaultdict

json_files = os.listdir('data/KodCode_json')
os.makedirs('data/KodCode_json_splitBySubset', exist_ok=True)

# 先收集所有数据，再一次性写入
subset_data = defaultdict(list)

for file in json_files:
    print(f"处理文件: {file}")
    try:
        with open(f'data/KodCode_json/{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            subset = item['gpt_difficulty']# subset,style,gpt_difficulty
            subset_data[subset].append(item)
            
    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")
os.makedirs('data/KodCode_json_splitByGptDifficulty', exist_ok=True)
# 一次性写入每个subset的文件
for subset, items in subset_data.items():
    output_file = f'data/KodCode_json_splitByGptDifficulty/{subset}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
    print(f"已创建 {output_file}, 包含 {len(items)} 条数据")

print("处理完成！")