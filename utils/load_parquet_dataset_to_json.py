# 将parquet文件转换为json文件
import pandas as pd
import json
import os
os.makedirs('data/KodCode_json', exist_ok=True)
for item in os.listdir('data/KodCode_parquet/data'):
    # 读取parquet文件
    df = pd.read_parquet(f'data/KodCode_parquet/data/{item}')

    # 将数据集转换为json文件
    df.to_json(f'data/KodCode_json/{item}.json', orient='records', lines=False,indent=4)