# -*- coding: utf-8 -*-
"""检查 LLM 新闻数据集结构 + 行业映射 + 价格数据可用性"""
import pandas as pd
import os, glob

root = r"C:\Users\liuqi\quant_system_v2"

# 1) all_news_stocks 的 impact 取值分布
df = pd.read_csv(os.path.join(root, "all_news_stocks.csv"))
print("=== all_news_stocks impact 分布 ===")
print(df["impact"].describe())
print(df["impact"].value_counts().head(15))
print(df.head(3).to_string())

# 2) news_price_merged 完整列
pm = pd.read_csv(os.path.join(root, "news_price_merged.csv"), nrows=5)
print("\n=== news_price_merged 列 ===")
print(list(pm.columns))

# 3) 行业映射文件
for pat in ["**/industry*.parquet", "**/stock_industry*.parquet"]:
    for f in glob.glob(os.path.join(root, pat), recursive=True):
        if "__pycache__" in f: continue
        try:
            t = pd.read_parquet(f)
            print(f"\n=== {os.path.relpath(f, root)}: {len(t)}行, 列={list(t.columns)[:10]}")
            print(t.head(2).to_string())
        except Exception as e:
            print(f"[读失败] {f}: {e}")
