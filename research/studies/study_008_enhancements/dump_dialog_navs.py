# -*- coding: utf-8 -*-
"""一次性脚本: 导出关键变体月度净值序列 (供对话内绘图, 与 risk_control_bt / direction2 同口径)"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements.direction2_hrp import _hrp_weights, WINDOW

TIER3_BND = [1.0, 0.98]
TIER3_W = [1.0, 0.5]
TIER5_BND = [1.0, 0.99, 0.98, 0.97]
TIER5_W = [1.0, 0.75, 0.5, 0.25]


def _ma20_w(c, m, tier):
    r = c / m
    if tier == "t3":
        return 1.0 if r >= 1.0 else (0.5 if r >= 0.98 else 0.0)
    w = 0.0
    for wgt, bnd in zip(TIER5_W, TIER5_BND):
        if r >= bnd:
            w = wgt
            break
    return w


def run(env, use_hrp, use_ma20, tier):
    navs = {}
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        nav = navs.get(rb, 1.0)
        if picks is None:
            navs[rb_next] = nav
            continue
        if use_hrp:
            hi = env.trade_dates.index(rb)
            win = env.trade_dates[max(0, hi - WINDOW):hi]
            rets = env.pct_df.reindex(columns=picks).reindex(win)
            w = _hrp_weights(rets)
        else:
            w = pd.Series(1.0 / len(picks), index=picks)
        cr = (comb * w.reindex(comb.columns)).sum(axis=1, min_count=1)
        for t in hold:
            r_t = e_ret.loc[t]
            if rs12_on:
                ww = 1.0
                if use_ma20:
                    c = env.idx_close_1.get(t, np.nan)
                    m = env.ma20_1.get(t, np.nan)
                    if np.isfinite(c) and np.isfinite(m):
                        ww = _ma20_w(c, m, tier)
                r_t = ww * cr.loc[t]
            nav *= (1.0 + r_t)
        navs[rb_next] = nav * (1.0 - C.COST)
    return pd.Series(navs).sort_index()


def main():
    env = C.Env()
    out = {}
    for lb, use_hrp, use_ma20, tier in [
        ("BASE+VAL", False, False, "t3"),
        ("+MA20三档098", False, True, "t3"),
        ("+HRP", True, False, "t3"),
        ("+HRP+MA20三档098", True, True, "t3"),
        ("+HRP+MA20五档098", True, True, "t5"),
    ]:
        nav = run(env, use_hrp, use_ma20, tier)
        out[lb] = {k: round(float(v), 4) for k, v in nav.items()}
    # 基准 512100ETF 月度累计
    bm = {}
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if rb not in bm:
            bm[rb] = 1.0
        bm[rb_next] = bm[rb] * float((1 + e_ret.reindex(hold).fillna(0.0)).prod())
    out["512100ETF"] = {k: round(v, 4) for k, v in pd.Series(bm).sort_index().items()}
    fp = os.path.join(C.DATA_DIR, "dialog_navs.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved:", fp)
    for k, v in out.items():
        print(k, len(v), list(v.items())[0], list(v.items())[-1])


if __name__ == "__main__":
    main()
