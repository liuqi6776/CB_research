# -*- coding: utf-8 -*-
"""检查 super_dataset.parquet 是否可做价格面板(列名/日期范围/股票数)"""
import duckdb

con = duckdb.connect()
f = r"C:\Users\liuqi\quant_system_v2\daily-pro-t1\data\super_dataset.parquet"

cols = con.execute(f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{f}'))").df()
print("列名:", list(cols["column_name"])[:40])
print("总列数:", len(cols))

info = con.execute(f"""
    SELECT COUNT(*) AS rows,
           MIN(trade_date) AS min_d, MAX(trade_date) AS max_d,
           COUNT(DISTINCT ts_code) AS n_stocks
    FROM read_parquet('{f}')
""").df()
print(info.to_string())
