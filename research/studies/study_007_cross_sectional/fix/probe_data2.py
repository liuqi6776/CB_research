# -*- coding: utf-8 -*-
import pandas as pd
import os, glob

BASE = r'D:\iquant_data\data_v2'

# 1. income1 覆盖范围与结构
files = sorted(os.listdir(os.path.join(BASE, 'income1')))
print('income1 range:', files[0], '->', files[-1], 'count=', len(files))
df = pd.read_parquet(os.path.join(BASE, 'income1', files[-1]))
print('income1 cols:', df.columns.tolist())
print(df.head(3).to_string())
print()

# 2. industry1 全部文件
print('industry1 files:', os.listdir(os.path.join(BASE, 'industry1')))

# 3. 找股票->行业映射（含name/list_date）
for f in os.listdir(os.path.join(BASE, 'industry1')):
    p = os.path.join(BASE, 'industry1', f)
    if f.endswith('.parquet'):
        d = pd.read_parquet(p)
        print(f, d.shape, d.columns.tolist()[:12])

# 4. money1 / board1 / margin1 / motion1 / chouma1 快速看
for d in ['money1', 'board1', 'motion1']:
    pdir = os.path.join(BASE, d)
    fs = [x for x in os.listdir(pdir) if x.endswith('.parquet')]
    if fs:
        dd = pd.read_parquet(os.path.join(pdir, sorted(fs)[-1]))
        print(d, dd.shape, dd.columns.tolist()[:14])

# 5. 验证 pct_chg 是复权收益：找一个除息样本
day1 = os.path.join(BASE, 'data_day1')
icbc = []
for f in ['20230714.parquet', '20230717.parquet']:
    d = pd.read_parquet(os.path.join(day1, f))
    icbc.append(d[d.ts_code == '601398.SH'][['trade_date','close','pre_close','pct_chg']])
print(pd.concat(icbc).to_string())

# 6. 项目内是否有 stock_basic / 股票名称表
import glob as g
hits = []
for pat in [r'C:\Users\liuqi\quant_system_v2\**\stock*.parquet', r'C:\Users\liuqi\quant_system_v2\**\*basic*.parquet', r'C:\Users\liuqi\quant_system_v2\**\*name*.parquet']:
    hits += g.glob(pat, recursive=True)
print('stock basic candidates:', hits[:15])
