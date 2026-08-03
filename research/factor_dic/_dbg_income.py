# -*- coding: utf-8 -*-
import os
import pandas as pd

D = "D:/iquant_data/data_v2/income1"
for f in ["20200630.parquet", "20201231.parquet", "20220331.parquet"]:
    fp = os.path.join(D, f)
    if not os.path.exists(fp):
        print(f, "NO FILE")
        continue
    df = pd.read_parquet(fp)
    print(f, "cols:", df.columns.tolist()[:40])
    print("   n=", len(df), "head:", df.head(1).to_dict("records"))
    print()
