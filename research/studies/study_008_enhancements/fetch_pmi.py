# -*- coding: utf-8 -*-
"""保存 iFinD EDB 制造业PMI 月度序列 (2020-01~2026-06, 来源: 国家统计局/iFinD EDB)
另探测 moneyflow1 资金流数据质量 (方向A 因子构造依据)
"""
import os
import glob
import numpy as np
import pandas as pd

BASE = "D:/iquant_data/data_v2"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---- 1. PMI (iFinD EDB 三次查询合并, 完整月度) ----
pmi = {
    "2020-01": 50.0, "2020-02": 35.7, "2020-03": 52.0, "2020-04": 50.8,
    "2020-05": 50.6, "2020-06": 50.9, "2020-07": 51.1, "2020-08": 51.0,
    "2020-09": 51.5, "2020-10": 51.4, "2020-11": 52.1, "2020-12": 51.9,
    "2021-01": 51.3, "2021-02": 50.6, "2021-03": 51.9, "2021-04": 51.1,
    "2021-05": 51.0, "2021-06": 50.9, "2021-07": 50.4, "2021-08": 50.1,
    "2021-09": 49.6, "2021-10": 49.2, "2021-11": 50.1, "2021-12": 50.3,
    "2022-01": 50.1, "2022-02": 50.2, "2022-03": 49.5, "2022-04": 47.4,
    "2022-05": 49.6, "2022-06": 50.2, "2022-07": 49.0, "2022-08": 49.4,
    "2022-09": 50.1, "2022-10": 49.2, "2022-11": 48.0, "2022-12": 47.0,
    "2023-01": 50.1, "2023-02": 52.6, "2023-03": 51.9, "2023-04": 49.2,
    "2023-05": 48.8, "2023-06": 49.0, "2023-07": 49.3, "2023-08": 49.7,
    "2023-09": 50.2, "2023-10": 49.5, "2023-11": 49.4, "2023-12": 49.0,
    "2024-01": 49.2, "2024-02": 49.1, "2024-03": 50.8, "2024-04": 50.4,
    "2024-05": 49.5, "2024-06": 49.5, "2024-07": 49.4, "2024-08": 49.1,
    "2024-09": 49.8, "2024-10": 50.1, "2024-11": 50.3, "2024-12": 50.1,
    "2025-01": 49.1, "2025-02": 50.2, "2025-03": 50.5, "2025-04": 49.0,
    "2025-05": 49.5, "2025-06": 49.7, "2025-07": 49.3, "2025-08": 49.4,
    "2025-09": 49.8, "2025-10": 49.0, "2025-11": 49.2, "2025-12": 50.1,
    "2026-01": 49.3, "2026-02": 49.0, "2026-03": 50.4, "2026-04": 50.3,
    "2026-05": 50.0, "2026-06": 50.3,
}
s = pd.Series(pmi, name="pmi")
s.index.name = "month"
s.to_frame().to_csv(os.path.join(DATA_DIR, "pmi_monthly.csv"), encoding="utf-8")
print("PMI 已保存:", len(s), "个月,", s.index[0], "~", s.index[-1])

# ---- 2. moneyflow1 质量探测 ----
mf = glob.glob(os.path.join(BASE, "moneyflow1", "*.parquet"))
print("\nmoneyflow1 文件数:", len(mf))
# 抽样 3 天看覆盖率与字段
import random
random.seed(42)
samples = sorted(random.sample(mf, 3))
tot_codes, tot_nan = [], []
for fp in samples:
    df = pd.read_parquet(fp)
    d = os.path.basename(fp)[:8]
    # 超大单净流入可用率
    elg = (df["buy_elg_amount"] - df["sell_elg_amount"])
    denom = (df["buy_elg_amount"] + df["sell_elg_amount"] + df["buy_lg_amount"] + df["sell_lg_amount"])
    ok = denom > 0
    print(f"{d}: 股票数={len(df)}, 超大单/主力成交额>0 占比={ok.mean():.2%}, "
          f"net_mf 缺失={df['net_mf_amount'].isna().mean():.2%}")
    tot_codes.append(len(df))

# 与中证1000 成分匹配 (取一个调仓日)
iw = os.path.join(BASE, "index_weight", "iw_20240131.parquet")
if os.path.exists(iw):
    w = pd.read_parquet(iw)
    codes = set(w.iloc[:, 0].astype(str))
    mf_fp = os.path.join(BASE, "moneyflow1", "20240131.parquet")
    if os.path.exists(mf_fp):
        mf_df = pd.read_parquet(mf_fp)
        mf_codes = set(mf_df["ts_code"].astype(str))
        inter = codes & mf_codes
        print(f"\n中证1000(20240131): {len(codes)} 只, moneyflow1 当日: {len(mf_codes)} 只, 匹配: {len(inter)} ({len(inter)/max(len(codes),1):.1%})")
