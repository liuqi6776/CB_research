# -*- coding: utf-8 -*-
"""数据能力探测：决定修复方案的数据基础"""
import pandas as pd
import os
import glob

BASE = r'D:\iquant_data\data_v2'

def show(name, df, n=2):
    print(f"== {name} shape={df.shape}")
    print(df.columns.tolist())
    print(df.head(n).to_string())
    print()

# 1. other_day1 / skill1 是否含复权因子、市值
for d in ['other_day1', 'skill1']:
    p = os.path.join(BASE, d, '20240102.parquet')
    if os.path.exists(p):
        show(d, pd.read_parquet(p))

# 2. 日线主数据
p = os.path.join(BASE, 'data_day1', '20240102.parquet')
if os.path.exists(p):
    show('data_day1', pd.read_parquet(p))

# 3. 财报缓存：ann_date 覆盖范围、每年覆盖股票数
fp = os.path.join(BASE, 'fundamental1', 'fina_indicator_cache.parquet')
f = pd.read_parquet(fp)
print('fina cols sample:', [c for c in f.columns][:30])
f['ann_date'] = pd.to_datetime(f['ann_date'], errors='coerce')
print('ann_date range:', f['ann_date'].min(), '->', f['ann_date'].max())
print('stocks:', f['ts_code'].nunique())
f['ann_year'] = f['ann_date'].dt.year
print(f.groupby('ann_year')['ts_code'].nunique())
print()

# 4. income1（利润表公告）结构
p = os.path.join(BASE, 'income1', '20240102.parquet')
if os.path.exists(p):
    show('income1', pd.read_parquet(p))

# 5. 行业快照（含名称、上市日期?）
ind = None
for cand in glob.glob(os.path.join(BASE, 'industry1', '*.parquet'))[:5]:
    print('industry file:', cand)
    ind = pd.read_parquet(cand)
    show('industry1', ind)
    break

# 6. 日线覆盖时间范围
files = sorted(os.listdir(os.path.join(BASE, 'data_day1')))
print('data_day1 range:', files[0], '->', files[-1], 'count=', len(files))

# 7. 退市股检查：2020 初存在但 2025 末消失的股票数
d20 = pd.read_parquet(os.path.join(BASE, 'data_day1', '20200102.parquet'))
last_files = [x for x in files if x.startswith('202512')] or files[-20:]
d25 = pd.read_parquet(os.path.join(BASE, 'data_day1', last_files[-1]))
gone = set(d20['ts_code']) - set(d25['ts_code'])
print('stocks 2020-01-02:', d20['ts_code'].nunique(), 'gone by end:', len(gone))
