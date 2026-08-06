# -*- coding: utf-8 -*-
"""逐年分解: 推荐配置 vs 中证1000 (全段, 真实口径)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.risk_control_real import TIER5

env = C.Env()
td = env.trade_dates
open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)

# 推荐配置全段
s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                       e_ovn, e_intra, use_hrp=True, use_ma20=True, tier=TIER5,
                       dd_stop=0.10, dd_floor=0.15, stop_w=0.5, floor_w=0.5, recov=0.05)
nav = s
# 无风控 (HRP 无MA20无DD)
s0, _ = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                       e_ovn, e_intra, use_hrp=True, use_ma20=False)
nav0 = s0

# 中证1000 指数日收益 (parquet 降序存储, 先升序)
idx = pd.read_parquet(os.path.join(rv.IDX_DIR, "000852.SH.parquet"))
ix = idx.set_index("trade_date")["close"].astype(float).sort_index()
ix = ix[ix.index >= nav.index[0]]
ix_nav = ix / ix.iloc[0]

rows = []
years = sorted(set(nav.index.str[:4]))
for y in years:
    m = nav.index.str[:4] == y
    m0 = nav0.index.str[:4] == y
    mi = ix_nav.index.str[:4] == y
    r_ps = nav[m].iloc[-1] / nav[m].iloc[0] - 1 if nav[m].sum() else np.nan
    r_p0 = nav0[m0].iloc[-1] / nav0[m0].iloc[0] - 1 if nav0[m0].sum() else np.nan
    r_ix = ix_nav[mi].iloc[-1] / ix_nav[mi].iloc[0] - 1 if ix_nav[mi].sum() else np.nan
    rows.append((y, r_p0, r_ps, r_ix))

print(f"{'年份':<6}{'无风控':>10}{'推荐配置':>10}{'中证1000':>10}{'超额(推荐)':>12}")
lines = ["逐年分解 (真实口径)", "=" * 50,
         f"{'年份':<6}{'无风控':>10}{'推荐配置':>10}{'中证1000':>10}{'超额(推荐)':>12}"]
for y, r0, rs, rx in rows:
    ex = rs - rx if pd.notna(rs) and pd.notna(rx) else np.nan
    print(f"{y:<6}{r0:>10.1%}{rs:>10.1%}{rx:>10.1%}{ex:>12.1%}")
    lines.append(f"{y:<6}{r0:>10.1%}{rs:>10.1%}{rx:>10.1%}{ex:>12.1%}")
# 分两段
for lo, hi, tag in [("2020", "2022", "2020-2022(定参段)"), ("2023", "2026", "2023-2026(OOS段)")]:
    m = (nav.index >= f"{lo}0101") & (nav.index <= f"{hi}1231")
    m0 = (nav0.index >= f"{lo}0101") & (nav0.index <= f"{hi}1231")
    mi = (ix_nav.index >= f"{lo}0101") & (ix_nav.index <= f"{hi}1231")
    r_ps = nav[m].iloc[-1] / nav[m].iloc[0] - 1
    r_p0 = nav0[m0].iloc[-1] / nav0[m0].iloc[0] - 1
    r_ix = ix_nav[mi].iloc[-1] / ix_nav[mi].iloc[0] - 1
    yrs = (int(nav[m].index[-1][:4]) - int(nav[m].index[0][:4])) + (
        int(nav[m].index[-1][4:6]) - int(nav[m].index[0][4:6])) / 12.0 or 1.0
    if yrs < 0.5:
        yrs = 0.5
    line = (f"{tag} ({(nav[m].index[0])[:4]}-{(nav[m].index[-1])[:4]})  "
            f"累计: 无风控 {r_p0:+.1%} / 推荐 {r_ps:+.1%} / 指数 {r_ix:+.1%} / 超额 {r_ps - r_ix:+.1%}  "
            f"推荐年化 {(1 + r_ps) ** (1 / yrs) - 1:+.1%}")
    print(line)
    lines.append("")
    lines.append(line)

with open(os.path.join(C.OUT_DIR, "risk_control_yearly.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[saved] {C.OUT_DIR}/risk_control_yearly.txt")
