# -*- coding: utf-8 -*-
"""用 duckdb 探查: 原始新闻库 / industry.parquet / 价格缓存"""
import duckdb, os, glob

root = r"C:\Users\liuqi\quant_system_v2"
con = duckdb.connect()

# 1) 找所有 parquet / 大的 csv / db 文件(数据目录)
print("=== 数据类文件清单(>1MB) ===")
for pat in ["**/*.parquet", "**/*.db", "**/*.sqlite"]:
    for f in glob.glob(os.path.join(root, pat), recursive=True):
        if "__pycache__" in f or "node_modules" in f: continue
        sz = os.path.getsize(f) // (1024*1024)
        if sz >= 1:
            print(f"  {os.path.relpath(f, root)}  {sz}MB")

# 2) industry 映射
for f in [os.path.join(root, "stock_industry_map_cached.parquet")]:
    try:
        t = con.execute(f"SELECT * FROM read_parquet('{f}') LIMIT 3").df()
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{f}')").fetchone()[0]
        print(f"\n=== {os.path.basename(f)}: {n}行")
        print(t.to_string())
    except Exception as e:
        print(f"[读失败] {f}: {e}")

# 3) 找 industry.parquet 原文件
for f in glob.glob(os.path.join(root, "**/industry.parquet"), recursive=True):
    try:
        t = con.execute(f"SELECT * FROM read_parquet('{f}') LIMIT 3").df()
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{f}')").fetchone()[0]
        print(f"\n=== {os.path.relpath(f, root)}: {n}行")
        print(t.to_string())
    except Exception as e:
        print(f"[读失败] {f}: {e}")
