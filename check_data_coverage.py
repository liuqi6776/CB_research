# -*- coding: utf-8 -*-
"""盘点本地新闻/价格/行业数据的时段覆盖, 确定样本外验证数据可行性"""
import pandas as pd
import os

root = r"C:\Users\liuqi\quant_system_v2"

candidates = [
    "all_news_stocks.csv", "all_news_market.csv", "all_news_sectors.csv",
    "news_academic_full.csv", "news_stock_matched_v2.csv", "news_price_merged.csv",
    "sector_sentiment_factors.csv", "sector_sentiment.csv",
    "negative_reversal_analysis.csv", "news_stock_analysis.csv",
]
for f in candidates:
    p = os.path.join(root, f)
    if not os.path.exists(p):
        print(f"[缺失] {f}")
        continue
    try:
        df = pd.read_csv(p, nrows=200000)
        cols = list(df.columns)
        print(f"\n=== {f} ({os.path.getsize(p)//1024}KB, 列: {cols[:12]}{'...' if len(cols)>12 else ''})")
        # 找日期列
        date_col = None
        for c in cols:
            if "date" in c.lower() or "时间" in c or "time" in c.lower():
                date_col = c
                break
        if date_col:
            d = df[date_col].astype(str)
            print(f"    日期列[{date_col}]: min={d.min()} max={d.max()} rows={len(df)}")
        else:
            print(f"    rows={len(df)} (无日期列)")
    except Exception as e:
        print(f"[读取失败] {f}: {e}")
