# -*- coding: utf-8 -*-
"""PIT 数据核验 (3.24): 历史成分股是否含退市股 / 是否幸存者偏差 (new-tea-quant PIT 思路)

iw_*.parquet 为每月时点快照 (2020-01~2026-06, 每期 1000 只, 中证1000),
load_index_weight(rb) 取 <=rb 最近一期 -> PIT 口径。核验四层:

  1. 逐快照 PIT 即时有效性: 快照日 ±5 交易日内该成分股必须有行情
     (不存在"快照日尚未上市"或"快照日前已退市"的成分)
     - first_date > d+5  → 回填成分 (该时点尚未上市, 不可能在指数里 → 幸存者偏差)
     - last_date  < d-5  → 已退市/停牌成分 (该时点已无交易, 同样可疑)
  2. 历史快照 vs 最新快照成分重合度 — >95% 说明"当前成分回填"
  3. 全段调仓名单 union 中, 回测末期 (最近 60 交易日) 已无行情数据的退市/停牌股
  4. 判定

输出: results/survivorship_check.txt|json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements import common as C

OUT_TXT = os.path.join(C.OUT_DIR, "survivorship_check.txt")
OUT_JSON = os.path.join(C.OUT_DIR, "survivorship_check.json")
WIN = 5  # 快照日 ±WIN 个交易日视为"该时点有行情"


def main():
    iw_dates = sorted(f[3:11] for f in os.listdir(rv.IW_DIR) if f.startswith("iw_"))
    members = {}
    for d in iw_dates:
        df = pd.read_parquet(os.path.join(rv.IW_DIR, f"iw_{d}.parquet"))
        members[d] = set(df["con_code"].astype(str).str.strip())
    latest_d = iw_dates[-1]
    latest_m = members[latest_d]

    all_codes = sorted(set().union(*members.values()))
    td = C.Env().trade_dates
    stocks, pct_df, _, _, _ = rv.load_panels(td, all_codes, None)
    idx = list(pct_df.index)
    pos = {d: i for i, d in enumerate(idx)}
    last_day = idx[-1]
    first_v = pct_df.apply(lambda s: s.first_valid_index())
    last_v = pct_df.apply(lambda s: s.last_valid_index())

    lines = ["PIT 数据核验 (3.24): 历史成分股含退市股? / 幸存者偏差检查 (new-tea-quant PIT 思路)",
             "=" * 96]
    lines.append(f"指数权重快照: {len(iw_dates)} 期 ({iw_dates[0]} ~ {iw_dates[-1]}), "
                 f"每期 1000 只; 行情面板 {len(idx)} 交易日, 末日 {last_day}")
    lines.append("")

    # ---- [1] 逐快照 PIT 即时有效性 ----
    lines.append("[1] 逐快照成分 PIT 即时有效性 (快照日 ±%d 交易日内有行情):" % WIN)
    lines.append(f"    {'快照':<10}{'成分数':>6}{'有效':>8}{'回填(未上市)':>12}{'已退市/停牌':>12}")
    rows = []
    for d in iw_dates:
        m = members[d]
        p = pos.get(d)
        if p is None:
            # 快照日不在交易日面板 (月末 vs 交易日): 取最近一个 <=d 的交易日
            before = [dd for dd in idx if dd <= d]
            p = idx.index(before[-1]) if before else None
        w_lo = max(0, p - WIN)
        w_hi = min(len(idx), p + WIN + 1)
        win_dates = idx[w_lo:w_hi]
        sub = pct_df.reindex(index=win_dates, columns=sorted(m))
        alive = sub.notna().any(axis=0)
        first_in = pd.Series({c: first_v.get(c) for c in sorted(m)})
        last_in = pd.Series({c: last_v.get(c) for c in sorted(m)})
        d_after = (first_in > max(win_dates)).sum()          # 快照后才有数据 → 回填
        d_before = (last_in < min(win_dates)).sum()          # 快照前已无数据 → 已退市/停牌
        ok = int(alive.sum())
        rows.append(dict(d=d, n=len(m), ok=ok, retro=int(d_after), dead=int(d_before)))
    for r in rows[::8] + [rows[-1]]:
        lines.append(f"    {r['d']:<10}{r['n']:>6}{r['ok']:>8}{r['retro']:>12}{r['dead']:>12}")
    tot = pd.DataFrame(rows)
    retro_all = int(tot["retro"].sum())
    dead_snap = int(tot["dead"].sum())
    lines.append(f"    ── 全期累计: 回填(快照日未上市) {retro_all} 只次 | 快照日已退市/停牌 {dead_snap} 只次 "
                 f"(总 {len(tot)*1000} 只次)")

    # ---- [2] 历史 vs 最新重合 ----
    lines.append("")
    lines.append("[2] 各期快照 vs 最新快照成分重合度 (20200123 抽样):")
    ov_first = len(members[iw_dates[0]] & latest_m) / 1000.0
    ov_all = len(set().union(*members.values()) & latest_m) / len(latest_m)
    lines.append(f"    首期 {iw_dates[0]} 与最新重合: {ov_first:.1%}")
    lines.append(f"    全期 union ({len(set().union(*members.values()))} 只) 与最新重合: {ov_all:.1%}")

    # ---- [3] 末期退市/停牌股 ----
    lines.append("")
    cutoff = idx[-60]
    dead = last_v[last_v < cutoff].index
    lines.append("[3] 全段调仓名单 union 中的退市/长期停牌股 (末期 60 日无行情):")
    lines.append(f"    数量: {len(dead)} / {len(all_codes)} ({len(dead)/len(all_codes):.1%})")
    if len(dead):
        ddf = pd.DataFrame({
            "last_date": [last_v[c] for c in sorted(dead)],
            "in_hist_iw": [sum(1 for d in iw_dates if c in members[d]) for c in sorted(dead)],
        }, index=sorted(dead)).sort_values("last_date")
        lines.append("    名单 (code: 最后行情日, 成分期数):")
        for c, r in ddf.iterrows():
            lines.append(f"      {c}: {r['last_date']}, 成分期数 {r['in_hist_iw']}")
    alive_ratio = (last_v.notna() & (last_v >= cutoff)).mean()
    lines.append(f"    全期 union 末期存活率: {alive_ratio:.1%}")

    # ---- [4] 判定 ----
    lines.append("")
    n_snap_total = len(tot) * 1000
    dead_rate = dead_snap / n_snap_total
    if retro_all > 0:
        verdict = (f"⚠ 幸存者偏差疑点: 全期 {retro_all} 只次回填 (快照日尚未上市却出现在历史快照中), "
                   "需人工复核数据源")
    elif ov_first > 0.95:
        verdict = "疑似幸存者偏差: 首期与最新成分重合 >95%, 历史快照疑为当前成分回填"
    elif dead_rate > 0.005:
        verdict = (f"⚠ 边界疑点: {dead_snap} 只次 ({dead_rate:.2%}) 快照日已无行情, "
                   "多为快照边界停牌, 可接受但建议复核")
    else:
        verdict = (f"无幸存者偏差: 历史快照为真实 PIT 成分 (0 回填), 首期重合 {ov_first:.0%} 属指数自然换手, "
                   f"全段 {len(dead)} 只退市/停牌股 ({len(dead)/len(all_codes):.1%}) 均真实出现在历史快照中, "
                   "回测已如实承担其退市风险")
    lines.append(f"判定: {verdict}")
    print("\n".join(lines))

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(n_snap=len(iw_dates), snap_first=iw_dates[0], snap_last=iw_dates[-1],
                       retro_total=retro_all, dead_snap_total=dead_snap, dead_snap_rate=float(dead_rate),
                       ov_first=ov_first, ov_all=ov_all,
                       dead_codes=[str(c) for c in sorted(dead)],
                       dead_n=len(dead), alive_ratio=float(alive_ratio), verdict=verdict),
                  f, ensure_ascii=False, indent=1)
    print(f"[saved] {OUT_TXT} | .json")


if __name__ == "__main__":
    main()
