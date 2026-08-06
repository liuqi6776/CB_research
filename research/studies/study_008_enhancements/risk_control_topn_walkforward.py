# -*- coding: utf-8 -*-
"""Top N 多段 walk-forward 检查 (HRP 纯选股, 无 MA20/DD, 生产连续口径)

回答: 若把选股数从 Top50 换成 Top55 (或 Top60), 在 4 段滚动 OOS (2023~2026)
各段的收益/回撤/Sharpe 是否稳定? 累计生产曲线是否优于 Top50?

口径 (与 baseline_cmp 3.9 同): 全段连续回测 -> 按 OOS 窗口切片 (生产真实表现)
对比: Top50 (当前) / Top55 (候选) / Top60 (单段样本峰值) + 中证1000
输出: results/risk_control_topn_walkforward.txt
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E

KS = [50, 55, 60]
# 滚动 OOS 窗口 (与 risk_control_walkforward 的 4 段一致, 日历对齐)
OOS_WINS = [
    ("2023", "20230101", "20231231"),
    ("2024", "20240101", "20241231"),
    ("2025", "20250101", "20251231"),
    ("2026", "20260101", "20261231"),
]


def build_picks(env, k):
    """与 common.Env._build_picks 同逻辑, 仅 Top 数参数化"""
    picks_map = {}
    for rb in env.rebal:
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        fvals = {}
        for code in members:
            f1, f2, ft = env.ret_1m.get(code), env.ivol.get(code), env.turn.get(code)
            fr = env.fwd.get(code)
            if fr is None or rb not in fr.index:
                continue
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            for name in env.panels:
                p = env.panels[name].get(rb)
                if p is not None and code in p.index:
                    v = p.loc[code]
                    if np.isfinite(v):
                        row[name] = v
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < k:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = [c for c in sf.BASE_COLS + ["VAL"] if c in zdf.columns]
        has = zdf[cols].dropna()
        if len(has) < k:
            continue
        picks_map[rb] = has.mean(axis=1).nlargest(k).index.tolist()
    return picks_map


def win_stats(nav, w0, w1):
    m = nav[(nav.index >= w0) & (nav.index <= w1)]
    if len(m) < 2:
        return E.daily_stats(m), float("nan")
    return E.daily_stats(m), m.iloc[-1] / m.iloc[0] - 1.0


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)

    # 中证1000 指数净值 (与策略同起点)
    idx = pd.read_parquet(os.path.join(rv.IDX_DIR, "000852.SH.parquet"))
    ix = idx.set_index("trade_date")["close"].astype(float).sort_index()
    ix_nav = ix / ix.iloc[0]

    lines = ["Top N 多段 walk-forward 检查 (HRP 纯选股, 无风控, 生产连续口径)", "=" * 100]
    rows = {"k": {}, "index": {}}

    for k in KS:
        env.picks_map = build_picks(env, k)
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=True, use_ma20=False)
        full = E.daily_stats(s)
        oos_all = s[s.index >= OOS_WINS[0][1]]
        oos_m = E.daily_stats(oos_all)
        print(f"[Top{k}] 全段回测完成: 年化 {full['cagr']*100:.2f}% 累计 {s.iloc[-1]-1:+.1%}", flush=True)

        lines.append("")
        lines.append(f"Top{k}  全段 年化 {full['cagr']*100:5.2f}%  Sharpe {full['shp']:4.2f}  "
                     f"MaxDD {full['dd']*100:5.2f}%  累计 {s.iloc[-1]-1:+6.1%}  |  "
                     f"OOS合计 年化 {oos_m['cagr']*100:5.2f}%  Sharpe {oos_m['shp']:4.2f}  "
                     f"MaxDD {oos_m['dd']*100:5.2f}%  累计 {oos_all.iloc[-1]/oos_all.iloc[0]-1:+6.1%}")
        lines.append(f"  {'OOS段':<6}{'年化':>8}{'Sharpe':>7}{'MaxDD':>9}{'卡玛':>7}{'累计':>9}")

        krows = {}
        for label, w0, w1 in OOS_WINS:
            m, cum = win_stats(s, w0, w1)
            r_ix, ix_cum = None, None
            krows[label] = dict(cagr=m["cagr"], shp=m["shp"], dd=m["dd"], k=m["k"], cum=cum)
            lines.append(f"  {label:<6}{m['cagr']:>8.2%}{m['shp']:>7.2f}{m['dd']:>9.2%}{m['k']:>7.2f}{cum:>9.1%}")
        rows["k"][k] = dict(full=dict(cagr=full["cagr"], shp=full["shp"], dd=full["dd"],
                                      k=full["k"], cum=float(s.iloc[-1] - 1)),
                            oos_all=dict(cagr=oos_m["cagr"], shp=oos_m["shp"], dd=oos_m["dd"],
                                         k=oos_m["k"], cum=float(oos_all.iloc[-1] / oos_all.iloc[0] - 1)),
                            wins=krows)

    # 指数各 OOS 段 + OOS 合计
    lines.append("")
    lines.append("中证1000 基准:")
    idx_rows = {}
    oos_ix = ix_nav[ix_nav.index >= OOS_WINS[0][1]]
    lines.append(f"  OOS合计 累计 {oos_ix.iloc[-1]/oos_ix.iloc[0]-1:+6.1%}")
    for label, w0, w1 in OOS_WINS:
        _, cum = win_stats(ix_nav, w0, w1)
        idx_rows[label] = dict(cum=cum)
        lines.append(f"  {label:<6}{'':>16}{cum:>9.1%}")
    rows["index"] = idx_rows

    fp = os.path.join(C.OUT_DIR, "risk_control_topn_walkforward.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "risk_control_topn_walkforward.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n[saved] {fp}")


if __name__ == "__main__":
    main()
