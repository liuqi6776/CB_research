# -*- coding: utf-8 -*-
"""查看 features_longterm.parquet 的 schema"""
import pyarrow.parquet as pq
import os

p = r'c:\Users\liuqi\quant_system_v2\longterm-research\data\features_longterm.parquet'
f = pq.ParquetFile(p)
names = f.schema_arrow.names
print('行数:', f.metadata.num_rows)
print('列数:', len(names))
for i in range(0, len(names), 20):
    print(' | '.join(names[i:i+20]))
