# -*- coding: utf-8 -*-
"""对比 全段 vs 受限段 的 基线/账本 NAV (2024-07~10), 定位分歧"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements import ledger as L

env = C.Env()
td = env.trade_dates
open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)
etf = C.load_idx("512100.SH")
e_open_s = etf["open"].astype(float)
e_close_s = etf["close"].astype(float)

# 全段
sb, _ = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                       e_ovn, e_intra, use_hrp=True, use_ma20=False)
sl, _ = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s)
# 受限段
sbr, _ = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                        e_ovn, e_intra, use_hrp=True, use_ma20=False,
                        start="20240101")
slr, _ = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s,
                      start="20240101")

days = [t for t in sb.index if "20240701" <= t <= "20241015"]
print(f"{'日期':<10}{'全段基线':>10}{'全段账本':>10}{'全段比':>9}{'受限基线':>10}{'受限账本':>10}{'受限比':>9}")
for t in days:
    print(f"{t:<10}{sb.loc[t]:>10.4f}{sl.loc[t]:>10.4f}{sl.loc[t]/sb.loc[t]-1:>+9.3%}"
          f"{sbr.loc[t]:>10.4f}{slr.loc[t]:>10.4f}{slr.loc[t]/sbr.loc[t]-1:>+9.3%}")
