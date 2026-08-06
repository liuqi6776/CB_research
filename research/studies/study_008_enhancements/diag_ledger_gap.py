# -*- coding: utf-8 -*-
"""诊断: 双账本 vs 基线 +12% 偏差的来源 — 是否调仓日隔夜跳空 (旧仓 rb收盘→t0开盘)
假设: engine 在调仓日 j==0 只计日内段 (V_close[0]/V_open[0]-1), 漏掉旧持仓
      在 t0 开盘前夜的隔夜收益; ledger 按绝对价格盯市自然计入该段.
验证: 逐月比较 ledger/engine 月收益差, 与 旧权重×隔夜收益 是否吻合
"""
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
                           e_ovn, e_intra, use_hrp=True, use_ma20=False)
s_led, _ = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s)

# 月频收益 (按调仓月切)
mseg = list(env.month_segments())
base_ret, led_ret, gap_est = {}, {}, {}
for rb, rb_next, hold, picks, comb, e_ret, rs12_on in mseg:
    if not len(hold):
        continue
    t0 = hold[0]
    if t0 not in s_base.index or t0 not in s_led.index:
        continue
    h_end = hold[-1]
    base_ret[rb] = s_base.loc[h_end] / s_base.loc[t0]  # 月内
    led_ret[rb] = s_led.loc[h_end] / s_led.loc[t0]
    # 估算: 该月调仓日 t0 的旧仓隔夜 (rb收盘 → t0开盘) — 用前月持仓近似省略, 仅打印差异
# 直接看月初净值比是否跳变
rb0 = list(base_ret)[0]
print(f"{'调仓月':<10}{'基线月内收益':>14}{'账本月内收益':>14}{'月差':>10}")
for rb in list(base_ret)[:10]:
    b, l = base_ret[rb], led_ret[rb]
    print(f"{rb:<10}{b-1:>14.4%}{l-1:>14.4%}{(l-b):>10.4%}")

# 调仓日隔夜跳跃: 看 ledger 在 t0 当日 NAV vs 前一交易日的比值, 是否远大于纯日内
print("\n逐月累计比值 ledger/base (月内末值):")
lines = []
for rb, rb_next, hold, picks, comb, e_ret, rs12_on in mseg:
    if not len(hold) or hold[-1] not in s_base.index:
        continue
    b = s_base.loc[hold[-1]]
    l = s_led.loc[hold[-1]]
    lines.append(f"{rb}: {l/b-1:+.4%}")
# 每6个一行压缩输出
for i in range(0, len(lines), 6):
    print("  " + " | ".join(lines[i:i+6]))
