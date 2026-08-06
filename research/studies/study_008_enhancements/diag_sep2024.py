# -*- coding: utf-8 -*-
"""诊断 20240930 分歧: 逐日对比 ledger/base NAV 路径, 定位跳变日与个股"""
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

s_base, _ = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                           e_ovn, e_intra, use_hrp=True, use_ma20=False,
                           start="20240601", end="20241231")
s_led, _ = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s,
                        start="20240601", end="20241231")

seg_days = [t for t in s_base.index if t.startswith("202409")]
print("日期        基线NAV    账本NAV    比值")
for t in seg_days:
    print(f"{t}  {s_base.loc[t]:.4f}  {s_led.loc[t]:.4f}  {s_led.loc[t]/s_base.loc[t]-1:+.4%}")

# 20240830 调仓月: 目标池与权重 (信号 20240830, t0=20240902)
mseg = list(env.month_segments())
for rb, rb_next, hold, picks, comb, e_ret, rs12_on in mseg:
    if rb == "20240830":
        print(f"\n调仓月 {rb} hold={hold[0]}~{hold[-1]} rs12_on={rs12_on} n_picks={len(picks)}")
        t0 = hold[0]
        hi = td.index(rb)
        win = td[max(0, hi - 120):hi]
        rets = env.pct_df.reindex(columns=picks).reindex(win)
        from research.studies.study_008_enhancements.direction2_hrp import _ivw_weights
        w = _ivw_weights(rets)
        # 检查 t0 开盘价是否 0/NaN (引擎 replace(0, nan) 会丢弃, 账本会停牌阻塞)
        z = []
        for c in w.index:
            v = open_df.at[t0, c]
            if pd.isna(v) or v <= 0:
                z.append((c, v))
        print(f"t0={t0} 开盘价0/NaN 股票 {len(z)}: {z[:10]}")
        # ETF 腿?
        break
