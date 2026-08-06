# -*- coding: utf-8 -*-
"""P1-① 收尾: 多段 walk-forward 生产化验证 (滚动3年定参 -> 次年 OOS)

模拟真实部署流程: 每段用"当时可用的最近3年"网格定参(含 no-DD 候选), 次年独立 OOS。
4 个滚动窗口, 各段 OOS 拼接为连续生产曲线, 对照:
  - 无风控 (HRP 权重, 无 MA20 无 DD)
  - 仅五档098 (无 DD)
  - 固定推荐 DD(10,15) 回升5%
  - 滚动定参 (自适应)
回答:
  Q1 滚动定参是否优于固定推荐 (参数自适应是否有生产价值)?
  Q2 每段选出的最优参数漂移是否严重 (参数稳定性)?
  Q3 walk-forward 累计 (生产可实现) vs 全段定参 (含前视) 的差距?
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.risk_control_real import TIER5

# (定参起, 定参止, OOS起, OOS止)
WINDOWS = [
    ("20200101", "20221231", "20230101", "20231231"),
    ("20210101", "20231231", "20240101", "20241231"),
    ("20220101", "20241231", "20250101", "20251231"),
    ("20230101", "20251231", "20260101", "20261231"),
]
DD_GRID = [(0.08, 0.12), (0.09, 0.14), (0.10, 0.15), (0.12, 0.18)]
RECOVS = [0.02, 0.03, 0.05]


def concat_segments(segs):
    """各段净值拼接为连续曲线 (段间相乘)"""
    parts, mult = [], 1.0
    for seg in segs:
        if seg is None or len(seg) == 0:
            continue
        s = seg * mult
        parts.append(s)
        mult = s.iloc[-1]
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)

    def run(dd=None, recov=None, use_ma20=True, start=None, end=None):
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=True, use_ma20=use_ma20, tier=TIER5,
                               dd_stop=(dd[0] if dd else None), dd_floor=(dd[1] if dd else None),
                               stop_w=0.5, floor_w=0.5, recov=recov,
                               start=start, end=end)
        return s, st, E.daily_stats(s)

    lines = ["P1-① 收尾: 多段 walk-forward (滚动3年定参 -> 次年OOS, 五档098底座, 真实口径)", "=" * 100]
    rows = {"windows": [], "cum": {}}
    segs = {"plain": [], "nodd": [], "fixed": [], "wf": []}
    picks = {}  # 每段前段最优参数

    for i, (is0, is1, oos0, oos1) in enumerate(WINDOWS):
        label = f"W{i+1} OOS {oos0[:4]}"
        print(f"\n===== {label} (定参 {is0[:4]}-{is1[:4]}, OOS {oos0[:4]}-{oos1[:4]}) =====", flush=True)
        # ---- 前段定参网格 (含 no-DD) ----
        grid = {}
        s0, _, m0 = run(start=is0, end=is1)          # no-DD
        grid[None] = m0
        for ds, df_ in DD_GRID:
            for rec in RECOVS:
                _, _, m = run(dd=(ds, df_), recov=rec, start=is0, end=is1)
                grid[(ds, df_, rec)] = m
        best = max(grid, key=lambda k: grid[k]["k"])
        best_m = grid[best]
        best_desc = "无DD" if best is None else f"DD({int(best[0]*100)},{int(best[1]*100)})回升{int(best[2]*100)}%"
        picks[label] = best
        print(f"  前段最优: {best_desc} (卡玛 {best_m['k']:.2f}, 年化 {best_m['cagr']:.2%})")
        # ---- OOS 对照 ----
        oos = {}
        s, st, m = run(use_ma20=False, start=oos0, end=oos1)
        oos["无风控"] = m; segs["plain"].append(s)
        s, st, m = run(start=oos0, end=oos1)
        oos["仅五档098"] = m; segs["nodd"].append(s)
        s, st, m = run(dd=(0.10, 0.15), recov=0.05, start=oos0, end=oos1)
        oos["固定DD(10,15)回升5%"] = m; segs["fixed"].append(s)
        if best is None:
            s, st, m = run(start=oos0, end=oos1)
        else:
            s, st, m = run(dd=(best[0], best[1]), recov=best[2], start=oos0, end=oos1)
        oos[f"滚动定参({best_desc})"] = m; segs["wf"].append(s)
        rows["windows"].append(dict(label=label, best=best_desc, oos=oos))
        print(f"  OOS: " + " | ".join(f"{k} 卡玛{m['k']:.2f}(年化{m['cagr']:.1%})" for k, m in oos.items()))

    # ---- 累计 OOS 曲线 ----
    lines.append("")
    lines.append("累计 OOS 曲线 (2023-01 ~ 2026-07, 生产可实现口径):")
    cum = {}
    for lb, ss in segs.items():
        nav = concat_segments(ss)
        m = E.daily_stats(nav)
        cum[lb] = dict(cagr=m["cagr"], dd=m["dd"], k=m["k"], shp=m["shp"], n=len(nav))
        lines.append(f"  {lb:<14} 年化 {m['cagr']:6.2%}  Sharpe {m['shp']:5.2f}  MaxDD {m['dd']:6.2%}  卡玛 {m['k']:.2f}  (OOS {len(nav)} 天)")
        print(f"[cum] {lb:<14} 年化 {m['cagr']:.2%}  卡玛 {m['k']:.2f}")

    # 参数漂移统计
    lines.append("")
    lines.append("前段定参漂移: " + " | ".join(f"{k}: {v if v is None else f'DD({int(v[0]*100)},{int(v[1]*100)})回升{int(v[2]*100)}%'}" for k, v in picks.items()))
    no_dd_cnt = sum(1 for v in picks.values() if v is None)
    lines.append(f"  4 段中选 no-DD 的次数: {no_dd_cnt}")

    with open(os.path.join(C.OUT_DIR, "risk_control_walkforward.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "risk_control_walkforward.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, cum=cum, picks={k: (None if v is None else list(v)) for k, v in picks.items()}),
                  f, ensure_ascii=False, indent=1, default=str)
    print(f"\n[saved] {C.OUT_DIR}/risk_control_walkforward.txt")


if __name__ == "__main__":
    main()
