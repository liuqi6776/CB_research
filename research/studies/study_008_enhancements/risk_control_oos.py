# -*- coding: utf-8 -*-
"""P1-① 五档 + DD 参数独立样本验证 (前段定参 / 后段 OOS)

审查意见 #2 (参数同段数据挖掘) 的真实口径复核:
  - 前段 In-Sample: 2020-01 ~ 2023-12 (48 调仓月) 网格定参
  - 后段 OOS:      2024-01 ~ 2026-07 (31 调仓月) 独立验证
定参网格:
  MA20 结构: 三档098 / 五档098 / 五档097
  DD 参数:   dd_pairs x 谷底回升 (沿用 risk_control_ddstop 网格)
验证问题:
  Q1 前段最优参数组合在后段是否仍带来正增量 (vs 无风控)?
  Q2 前段参数排序与后段排序是否稳定 (Top 参数在后段不塌)?
  Q3 固定推荐 (五档098 + DD(10,15)回升5%) 是否被前段定参结果显著超越或打脸?
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.risk_control_real import TIER3, TIER5

IS_START, IS_END = "20200101", "20231231"
OOS_START, OOS_END = "20240101", "20261231"

TIER5_097 = {"bnd": [1.0, 0.99, 0.98, 0.97, 0.96], "w": [1.0, 0.8, 0.6, 0.4, 0.2]}
TIERS = {"三档098": TIER3, "五档098": TIER5, "五档097": TIER5_097}
DD_PAIRS = [(0.08, 0.12), (0.09, 0.14), (0.10, 0.15), (0.12, 0.18)]
RECOVS = [0.02, 0.03, 0.05]


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)

    def run(tag, tier, dd=None, recov=None, start=None, end=None, use_ma20=True):
        if dd is None:
            s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                   e_ovn, e_intra, use_hrp=True, use_ma20=use_ma20, tier=tier,
                                   start=start, end=end)
            return s, st, E.daily_stats(s)
        ds, df_ = dd
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=True, use_ma20=True, tier=tier,
                               dd_stop=ds, dd_floor=df_, stop_w=0.5, floor_w=0.5, recov=recov,
                               start=start, end=end)
        return s, st, E.daily_stats(s)

    lines = []
    results = {}

    # ---------- 前段 IS 网格定参 ----------
    print("=" * 100)
    print(f"前段 In-Sample 网格 ({IS_START}~{IS_END})")
    print("=" * 100)
    grid = {}
    # MA20 结构定档 (无 DD)
    for tn, tier in TIERS.items():
        s, st, m = run(tn, tier, start=IS_START, end=IS_END)
        grid[("MA20", tn, None, None)] = m
        print(f"  MA20 {tn:<6} 年化 {m['cagr']:6.2%}  Sharpe {m['shp']:5.2f}  日频MaxDD {m['dd']:6.2%}  卡玛 {m['k']:.2f}")
    best_tier_name = max(TIERS, key=lambda tn: grid[("MA20", tn, None, None)]["k"])
    print(f"  -> 前段最优 MA20 结构: {best_tier_name} (卡玛 {grid[('MA20', best_tier_name, None, None)]['k']:.2f})")
    # 最优档位下 DD 网格
    for ds, df_ in DD_PAIRS:
        for rec in RECOVS:
            key = ("DD", best_tier_name, (ds, df_), rec)
            s, st, m = run(best_tier_name, TIERS[best_tier_name], dd=(ds, df_), recov=rec,
                           start=IS_START, end=IS_END)
            grid[key] = m
            print(f"  +DD({int(ds*100)},{int(df_*100)}) 回升{int(rec*100)}%  年化 {m['cagr']:6.2%}  MaxDD {m['dd']:6.2%}  卡玛 {m['k']:.2f}")
    is_best = max((k for k in grid if k[0] == "DD"), key=lambda k: grid[k]["k"])
    _, btier, bdd, brec = is_best
    print(f"  -> 前段最优: MA20 {btier} + DD({int(bdd[0]*100)},{int(bdd[1]*100)}) 回升{int(brec*100)}%  (卡玛 {grid[is_best]['k']:.2f})")

    # ---------- 后段 OOS 验证 ----------
    print("=" * 100)
    print(f"后段 OOS ({OOS_START}~{OOS_END})")
    print("=" * 100)
    oos_rows = {}
    # 对照: 无风控 (HRP 权重, 无 MA20 无 DD) / 仅五档 / 固定推荐 / 前段最优 / 前段次优
    s, st, m0 = run("OOS无风控(BASE+VAL+HRP)", TIER5, start=OOS_START, end=OOS_END, use_ma20=False)
    oos_rows["无风控"] = (m0, st)
    print(f"  {'变体':<42}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'卡玛':>7}")
    print(f"  {'无风控 (HRP 权重)':<42}{m0['cagr']:>8.2%}{m0['shp']:>8.2f}{m0['dd']:>9.2%}{m0['k']:>7.2f}")

    def _oos(lb, tier, dd=None, recov=None):
        s, st, m = run(lb, tier, dd=dd, recov=recov, start=OOS_START, end=OOS_END)
        oos_rows[lb] = (m, st)
        print(f"  {lb:<42}{m['cagr']:>8.2%}{m['shp']:>8.2f}{m['dd']:>9.2%}{m['k']:>7.2f}")

    _oos("仅 MA20 五档098 (无 DD)", TIER5)
    _oos("固定推荐 五档098+DD(10,15)回升5%", TIER5, (0.10, 0.15), 0.05)
    _oos(f"前段最优 {btier}+DD({int(bdd[0]*100)},{int(bdd[1]*100)})回升{int(brec*100)}%",
         TIERS[btier], bdd, brec)

    # 参数排序稳定性: 前段卡玛 Top5 组合在后段的卡玛排名
    print("\n参数排序稳定性 (前段 Top5 → 后段):")
    top5 = sorted((k for k in grid if k[0] == "DD"), key=lambda k: grid[k]["k"], reverse=True)[:5]
    oos_grid = {}
    for k in top5:
        _, tn, dd, rec = k
        s, st, m = run(tn, TIERS[tn], dd=dd, recov=rec, start=OOS_START, end=OOS_END)
        oos_grid[k] = m
        print(f"  前段卡玛 {grid[k]['k']:.2f} ({tn}, DD({int(dd[0]*100)},{int(dd[1]*100)}), 回升{int(rec*100)}%)"
              f"  -> 后段卡玛 {m['k']:.2f}  年化 {m['cagr']:6.2%}  MaxDD {m['dd']:6.2%}")
    oos_rank = sorted(oos_grid, key=lambda k: oos_grid[k]["k"], reverse=True)
    print(f"  前段最优 {top5[0]} 在后段排名 {oos_rank.index(top5[0]) + 1}/{len(oos_rank)}")

    # ---------- 汇总 ----------
    summary = dict(is_best=[btier, bdd, brec], grid={str(k): v for k, v in grid.items()},
                   oos={k: v[0] for k, v in oos_rows.items()},
                   oos_rank=[str(k) for k in oos_rank])
    with open(os.path.join(C.OUT_DIR, "risk_control_oos.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)

    # 文本输出
    with open(os.path.join(C.OUT_DIR, "risk_control_oos.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[saved] {os.path.join(C.OUT_DIR, 'risk_control_oos.json')}")


if __name__ == "__main__":
    main()
