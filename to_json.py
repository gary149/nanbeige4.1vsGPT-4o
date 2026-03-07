import pandas as pd
# Convert train.parquet to jsonl
df = pd.read_parquet('train.parquet')
df.to_json('train.jsonl', orient='records', lines=True)
