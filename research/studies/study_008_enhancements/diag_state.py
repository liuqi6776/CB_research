# -*- coding: utf-8 -*-
"""诊断: 全段双账本状态在 2024-07~09 的切换 (无过滤), 定位 +10pp 来源"""
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

snaps = []

def debug(rb, snap):
    if "20240701" <= rb <= "20241001":
        snaps.append((rb, snap))

s_led, st = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s, debug=debug)
for rb, s in snaps:
    print(f"{rb}: t0={s['t0']} leg={'ETF' if s['use_etf'] else '股票'} n_units={s['n_units']} "
          f"cash={s['cash']:.5f} etf={s['etf']:.4f} pending={s['n_pending']} te={s['te']*100:.2f}%")

# 检查 20240830 调仓 (t0=20240902) 卖出侧: 上期股票持仓里停牌/无开盘价的
# 手动复刻: 取 20240731 月持仓 → 20240902 开盘价
print("\n检查 20240902 停牌持仓 (账本卖出阻塞候选):")
mseg = list(env.month_segments())
prev_units = None
for rb, rb_next, hold, picks, comb, e_ret, rs12_on in mseg:
    if rb == "20240731":
        # 复刻 ledger 在 20240731 月 (t0=20240801) 的买入
        t0 = hold[0]
        hi = td.index(rb)
        win = td[max(0, hi - 120):hi]
        rets = env.pct_df.reindex(columns=picks).reindex(win)
        from research.studies.study_008_enhancements.direction2_hrp import _ivw_weights
        w = _ivw_weights(rets)
        po = open_df.loc[t0]
        units = {c: float(w[c]) / po[c] for c in w.index if pd.notna(po[c]) and po[c] > 0}
        prev_units = units
        print(f"20240731 持仓 {len(units)} 只, 全为股票 (rs12_on={rs12_on})")
    if rb == "20240830" and prev_units is not None:
        t0 = hold[0]
        suspended = []
        for c in prev_units:
            v = open_df.at[t0, c]
            if pd.isna(v) or v <= 0:
                suspended.append((c, prev_units[c] * (close_df.at[td[td.index(t0)-1], c] if pd.notna(close_df.at[t0, c]) else np.nan)))
        wsum = sum(u * close_df.at[t0, c] for c, u in prev_units.items()
                   if pd.notna(open_df.at[t0, c]) and open_df.at[t0, c] > 0)
        print(f"20240902 (t0) 停牌/无开盘价持仓 {len(suspended)} 只: {suspended[:8]}")
        break
