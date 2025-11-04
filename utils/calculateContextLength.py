# 计算特定字段的平均长度，返回两个具体的长度值，能将所有数据分为三类：long，medium,short
import json
import os
from collections import defaultdict

json_files = os.listdir('data/KodCode_json')
length=[]
for file in json_files:
    print(f"处理文件: {file}")
    try:
        with open(f'data/KodCode_json/{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            length.append(len(item['solution'])+len(item['question']))
            
    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")

# 计算平均长度,以及三等分点，能将所有数据分为数据个数相等的三类：long，medium,short
average_length = sum(length) / len(length)
print(f"平均长度: {average_length}")

length=sorted(length)
third_point = length[len(length) // 3]
print(f"三等分点: {third_point}")