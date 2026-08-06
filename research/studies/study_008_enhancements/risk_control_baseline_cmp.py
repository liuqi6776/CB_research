# -*- coding: utf-8 -*-
"""剥离风控层对照验证 (3.8 结论的生产决策依据) — 全段回测 + 切片口径

注意: 独立子区间回测会跳过窗口首月持有期 (hold 从 rb 次日开始), 故采用
全段连续回测后按年切片 (策略从 2020 持续运行, 切片即生产真实表现)。
  口径: 等权Top60 / IVW120纯选股 / +三档098 / +五档098 / +五档098+DD(10,15)回升5%
  窗口: 全段 2020-2026 / 定参段 2020-2022 / OOS段 2023-2026
对照: 中证1000 指数
输出: results/risk_control_baseline_cmp.txt
"""
import os
import sys
import json

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.risk_control_real import TIER3, TIER5

# 全段窗口运行时取策略净值实际区间 (与指数同口径比较, 避免指数历史长于策略)
WINDOWS_FIXED = [("定参段", "20200101", "20221231"), ("OOS段", "20230101", "20261231")]
CASES = [
    ("等权Top60纯选股", dict(use_hrp=False, use_ma20=False)),
    ("IVW120纯选股", dict(use_hrp=True, use_ma20=False)),
    ("IVW120+三档098", dict(use_hrp=True, use_ma20=True, tier=TIER3)),
    ("IVW120+五档098", dict(use_hrp=True, use_ma20=True, tier=TIER5)),
    ("IVW120+五档098+DD(10,15)回升5%", dict(use_hrp=True, use_ma20=True, tier=TIER5,
                                            dd_stop=0.10, dd_floor=0.15, stop_w=0.5, floor_w=0.5, recov=0.05)),
]


def win_stats(nav, w0, w1):
    m = nav[(nav.index >= w0) & (nav.index <= w1)]
    if len(m) < 2:
        return E.daily_stats(m), float("nan")
    return E.daily_stats(m), m.iloc[-1] / m.iloc[0] - 1.0


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)

    # 指数净值 (升序)
    idx = pd.read_parquet(os.path.join(rv.IDX_DIR, "000852.SH.parquet"))
    ix = idx.set_index("trade_date")["close"].astype(float).sort_index()
    ix_nav = ix / ix.iloc[0]

    lines = ["剥离风控层对照验证 (真实口径, 全段回测+切片)", "=" * 100]
    rows = {"cases": {}, "index": {}}
    # 全段回测各口径
    navs = {}
    for lb, kw in CASES:
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, **kw)
        navs[lb] = s
        print(f"[backtest] {lb}", flush=True)
    # 全段窗口 = 策略净值实际区间 (2020 起), 保证与指数同口径
    full0 = str(navs[CASES[0][0]].index[0])
    full1 = str(navs[CASES[0][0]].index[-1])
    WINDOWS = [("全段", full0, full1)] + WINDOWS_FIXED
    for wname, w0, w1 in WINDOWS:
        lines.append("")
        lines.append(f"[{wname}] {'口径':<24}{'年化':>8}{'Sharpe':>7}{'MaxDD':>9}{'卡玛':>7}{'累计':>9}{'指数':>8}{'超额':>9}")
        print(f"\n[{wname}]", flush=True)
        _, r_ix = win_stats(ix_nav, w0, w1)
        rows["index"][wname] = dict(cum=r_ix)
        for lb in [c[0] for c in CASES]:
            m, cum = win_stats(navs[lb], w0, w1)
            ex = cum - r_ix
            rows["cases"].setdefault(lb, {})[wname] = dict(cagr=m["cagr"], shp=m["shp"], dd=m["dd"],
                                                           k=m["k"], cum=cum, ex=ex)
            lines.append(f"[{wname}] {lb:<24}{m['cagr']:>8.2%}{m['shp']:>7.2f}{m['dd']:>9.2%}{m['k']:>7.2f}"
                         f"{cum:>9.1%}{r_ix:>8.1%}{ex:>+9.1%}")
            print(f"  {lb:<26} 年化 {m['cagr']:.2%}  MaxDD {m['dd']:.2%}  卡玛 {m['k']:.2f}  累计 {cum:+.1%}  超额 {ex:+.1%}")

    with open(os.path.join(C.OUT_DIR, "risk_control_baseline_cmp.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "risk_control_baseline_cmp.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n[saved] {C.OUT_DIR}/risk_control_baseline_cmp.txt")


if __name__ == "__main__":
    main()
