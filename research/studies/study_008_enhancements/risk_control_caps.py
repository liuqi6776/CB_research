# -*- coding: utf-8 -*-
"""P1-② 单股 / 行业簇权重上限建模 (审查意见 #10)

真实口径 (+HRP +MA20五档098) 下:
  - 单股 cap 敏感性: 无约束(HRP 天然<=5%) / 3% / 2%
  - 行业簇 cap 敏感性: 无 / 30% / 25% / 20%  (东财行业映射)
输出: results/risk_control_caps.txt
"""
import os
import sys
import json

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.risk_control_real import TIER5


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)

    ind_map = {}
    fp = os.path.join(C.DATA_DIR, "industry_map.parquet")
    if os.path.exists(fp):
        im = pd.read_parquet(fp)
        ind_map = {str(r.ts_code): str(r.industry) for r in im.itertuples()}
        print(f"[industry_map] {len(ind_map)} 只")
    else:
        print("[industry_map] 缺失! 行业簇 cap 跳过")

    lines = ["P1-② 单股 / 行业簇上限 (真实口径 +HRP +MA20五档098)", "=" * 90,
             f"{'变体':<34}{'年化':>8}{'Sharpe':>8}{'日频MaxDD':>10}{'卡玛':>7}{'切换成本':>9}{'换手成本':>9}"]
    rows = {}
    cases = [
        ("基线(单股天然<=5%, 行业无上限)", None, None),
        ("单股cap3%", 0.03, None),
        ("单股cap2%", 0.02, None),
        ("行业簇cap30%", None, 0.30),
        ("行业簇cap25%", None, 0.25),
        ("行业簇cap20%", None, 0.20),
        ("单股cap3%+行业簇cap25%", 0.03, 0.25),
    ]
    for lb, cap, cap_ind in cases:
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=True, use_ma20=True, tier=TIER5,
                               cap=cap, cap_ind=cap_ind, ind_map=ind_map if cap_ind else None)
        m = E.daily_stats(s)
        rows[lb] = dict(**m, switch=st["switch"], turn=st["turn"])
        lines.append(f"{lb:<34}{m['cagr']:>8.2%}{m['shp']:>8.2f}{m['dd']:>10.2%}{m['k']:>7.2f}"
                     f"{st['switch']:>9.2%}{st['turn']:>9.2%}")
        print(f"  {lb:<34}{m['cagr']:>8.2%}  {m['dd']:>8.2%}  {m['k']:>6.2f}")

    # 行业簇权重分布: 基线 vs cap25% 在最近调仓日的最大行业权重
    lines.append("")
    lines.append("行业簇集中度 (20260731 调仓, HRP 权重):")
    import numpy as np
    from research.studies.study_008_enhancements.direction2_hrp import _hrp_weights, WINDOW
    from research.studies.study_008_enhancements.risk_control_real import cap_weights
    rb = "20260731"
    rb_next, hold, picks, comb, e_ret, rs12_on = None, None, None, None, None, None
    for a, b, c, d, e_, f_, g in env.month_segments():
        if a == rb:
            _, _, hold, picks, comb, e_ret, rs12_on = a, b, c, d, e_, f_, g
            break
    hi = td.index(rb)
    win = td[max(0, hi - WINDOW):hi]
    rets = env.pct_df.reindex(columns=picks).reindex(win)
    w = _hrp_weights(rets)
    ind_s = pd.Series({c: ind_map.get(c, "NA") for c in w.index})
    top = w.groupby(ind_s).sum().sort_values(ascending=False)
    lines.append(f"  基线 前5行业: " + ", ".join(f"{k} {v:.1%}" for k, v in top.head(5).items()))
    w2 = E.cap_industry(w.copy(), ind_map, 0.25)
    top2 = w2.groupby(pd.Series({c: ind_map.get(c, "NA") for c in w2.index})).sum().sort_values(ascending=False)
    lines.append(f"  cap25 前5行业: " + ", ".join(f"{k} {v:.1%}" for k, v in top2.head(5).items()))

    # 全期行业簇最大权重统计 (约束是否曾绑定)
    max_ind_w = []
    for a, b, c, d, e_, f_, g in env.month_segments():
        if d is None:
            continue
        hi = td.index(a)
        win = td[max(0, hi - WINDOW):hi]
        rets = env.pct_df.reindex(columns=d).reindex(win)
        ww = _hrp_weights(rets)
        ind_ss = pd.Series({cc: ind_map.get(cc, "NA") for cc in ww.index})
        max_ind_w.append(ww.groupby(ind_ss).sum().max())
    mx = pd.Series(max_ind_w)
    lines.append(f"  全期 {len(mx)} 个调仓日行业簇最大权重: max {mx.max():.1%}, 均值 {mx.mean():.1%}, "
                 f">20% 的月份 {int((mx > 0.20).sum())}, >25% 的月份 {int((mx > 0.25).sum())}")

    with open(os.path.join(C.OUT_DIR, "risk_control_caps.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "risk_control_caps.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {C.OUT_DIR}/risk_control_caps.txt")


if __name__ == "__main__":
    main()
