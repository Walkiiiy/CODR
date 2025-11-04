import json
import os
from collections import defaultdict

json_files = os.listdir('data/KodCode_json_splitByGptDifficulty')

for file in json_files:
    print(f"处理文件: {file}")
    try:
        with open(f'data/KodCode_json_splitByGptDifficulty/{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # 直接保留需要的字段，删除其他字段
            keys_to_keep = ['question_id', 'question', 'solution', 'test']
            keys_to_remove = [key for key in item.keys() if key not in keys_to_keep]
            for key in keys_to_remove:
                del item[key]
        os.makedirs('data/KodCode_splited/ByGptDifficulty/',exist_ok=True)     
        with open(f'data/KodCode_splited/ByGptDifficulty/{file}','w') as f:
            json.dump(data,f,indent=4)

    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")
        

print("处理完成！")