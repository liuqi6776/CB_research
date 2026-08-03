# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from config.settings import settings

print("fina_indicator_cache columns:")
fa = pd.read_parquet(os.path.join(settings.DATA_PATH, "fundamental1", "fina_indicator_cache.parquet"))
print(fa.columns.tolist())
print("rows:", len(fa), "sample ann_date:", fa["ann_date"].astype(str).str[:8].unique()[:3] if "ann_date" in fa.columns else "NA")

print("\nother_day1 sample file columns:")
d = sorted(os.listdir(os.path.join(settings.DATA_PATH, "other_day1")))[-1]
df = pd.read_parquet(os.path.join(settings.DATA_PATH, "other_day1", d))
print("file:", d, "cols:", df.columns.tolist(), "rows:", len(df))
