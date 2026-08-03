# -*- coding: utf-8 -*-
"""诊断: 新闻因子的各种聚合形态 vs 20d超额收益 的 IC/ICIR (2020-2026)"""
import pandas as pd
import numpy as np

p = r'c:\Users\liuqi\quant_system_v2\longterm-research\data\features_longterm.parquet'
df = pd.read_parquet(p, columns=['trade_date', 'ts_code', 'news_stock_impact',
                                 'news_market_impact', 'news_has_mention', 'mkt_excess_ret_20d'])
df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

# 滚动聚合新闻因子 (按股票)
g = df.groupby('ts_code')
df['news_ma5'] = g['news_stock_impact'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['news_ma20'] = g['news_stock_impact'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['news_ma60'] = g['news_stock_impact'].transform(lambda x: x.rolling(60, min_periods=1).mean())
df['news_diff'] = df['news_stock_impact'] - df['news_ma20']   # 当日-20日均值(新信息)
df['news_mention_ma5'] = g['news_has_mention'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['news_mkt_ma20'] = g['news_market_impact'].transform(lambda x: x.rolling(20, min_periods=1).mean())

cands = ['news_stock_impact', 'news_market_impact', 'news_has_mention',
         'news_ma5', 'news_ma20', 'news_ma60', 'news_diff', 'news_mention_ma5', 'news_mkt_ma20']

print(f"{'因子':<22}{'IC':>8}{'ICIR':>8}{'IC均值>0比例':>12}")
for c in cands:
    ics = df.groupby('trade_date').apply(lambda x: x[c].corr(x['mkt_excess_ret_20d'], method='spearman'))
    ics = ics.dropna()
    ic_mean = ics.mean()
    ic_std = ics.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    pos = (ics > 0).mean()
    print(f"{c:<22}{ic_mean:>8.4f}{icir:>8.3f}{pos:>12.2%}")
