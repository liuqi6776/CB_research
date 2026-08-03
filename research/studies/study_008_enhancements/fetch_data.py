# -*- coding: utf-8 -*-
"""方向1/3 数据准备: 北向资金历史 + 东财行业映射
- 北向: akshare stock_hsgt_hist_em('北向资金'), 2020-01~2024-08 有效(此后停止披露)
- 行业: tushare stock_basic (东财行业静态映射, 研究用近似)
输出: research/studies/study_008_enhancements/data/
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config.settings import settings

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 北向资金 ----------
try:
    import akshare as ak
    df = ak.stock_hsgt_hist_em(symbol="北向资金")
    df = df.rename(columns={"日期": "trade_date", "当日成交净买额": "north_net"})
    df["trade_date"] = df["trade_date"].astype(str).str[:10].str.replace("-", "")
    df = df[["trade_date", "north_net"]].dropna(subset=["north_net"])
    df = df[(df["trade_date"] >= "20200101") & (df["trade_date"] <= "20260801")]
    df.to_parquet(os.path.join(OUT, "north_flow.parquet"), index=False)
    print(f"north_flow: {len(df)} rows, {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}")
except Exception as e:
    print("north ERR:", str(e)[:200])

# ---------- 2. 东财行业映射 (tushare stock_basic) ----------
try:
    import tushare as ts
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    sb = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
    sb.to_parquet(os.path.join(OUT, "industry_map.parquet"), index=False)
    print(f"industry_map: {len(sb)} rows, 行业数={sb['industry'].nunique()}")
    print(sb["industry"].value_counts().head(15).to_string())
except Exception as e:
    print("industry ERR:", str(e)[:200])
