# -*- coding: utf-8 -*-
"""拉取中证1000成分股历史 namechange, 构建 ST 区间标记 (tushare, 断点续拉)
输出: data/st_history.parquet (ts_code, start_date, end_date, name, is_st)
"""
import os
import sys
import time

import pandas as pd
import tushare as ts

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config.settings import settings
from research.factor_dic import run_validation as rv

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "st_history.parquet")

# 全部历史成分股集合
all_codes = set()
trade_dates = rv.load_trade_dates()
months = {d[:6]: d for d in trade_dates if d[:4] >= "2020"}
rebal = sorted(months.values())
for rb in rebal:
    m = rv.load_index_weight(rb)
    if m:
        all_codes |= m
all_codes = sorted(all_codes)
print(f"成分股 {len(all_codes)} 只", flush=True)

pro = ts.pro_api(settings.TUSHARE_TOKEN)
have = set()
if os.path.exists(OUT):
    old = pd.read_parquet(OUT)
    have = set(old["ts_code"].unique())
    print(f"已缓存 {len(have)} 只, 待拉 {len(all_codes) - len(have)} 只", flush=True)

rows = []
fail = []
for i, code in enumerate(all_codes):
    if code in have:
        continue
    for attempt in range(3):
        try:
            df = pro.namechange(ts_code=code,
                                fields="ts_code,name,start_date,end_date,change_reason")
            if df is not None and not df.empty:
                rows.append(df)
            break
        except Exception as e:
            if attempt == 2:
                fail.append((code, str(e)))
            time.sleep(1.2)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(all_codes)}  已拉 {len(rows)}", flush=True)
        if rows:
            big = pd.concat(rows, ignore_index=True)
            big.to_parquet(OUT)
    time.sleep(0.3)

if rows:
    big = pd.concat(rows, ignore_index=True)
    if os.path.exists(OUT):
        big = pd.concat([pd.read_parquet(OUT), big], ignore_index=True)
    big = big.drop_duplicates(subset=["ts_code", "start_date", "name"], keep="last")
    big.to_parquet(OUT)
    print(f"[save] {len(big)} 行, {big['ts_code'].nunique()} 只", flush=True)
print(f"失败 {len(fail)} 只: {fail[:10]}", flush=True)
