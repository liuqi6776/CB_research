# -*- coding: utf-8 -*-
"""方向2 补充验证: HRP 参数敏感性 (窗口 60/120/180) + 分年 OOS
- 敏感性: WINDOW 变化下 HRP+MA20 vs 等权+MA20, 检验卡玛/MaxDD 增量是否稳健
- 分年 OOS: 按月拆分 2020~2026, 逐年对比 HRP+MA20 vs 等权+MA20 的年收益/回撤/卡玛
- 全部信号 T-1 已知、T 日生效, 与 direction2_hrp.py 同口径 (20bps, RS12, MA20三档0.98)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements.direction2_hrp import _hrp_weights, _get_cluster_sort

MA20_DEEP = 0.98


def run_hrp(env, use_hrp, use_ma20, window=120):
    """与 direction2_hrp.run_hrp 一致, 增加 window 参数"""
    navs = {}
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        nav = navs.get(rb, 1.0)
        if picks is None:
            navs[rb_next] = nav
            continue
        if use_hrp:
            hi = env.trade_dates.index(rb)
            win = env.trade_dates[max(0, hi - window):hi]
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
                        ww = 1.0 if c >= m else (0.5 if c >= MA20_DEEP * m else 0.0)
                r_t = ww * cr.loc[t]
            nav *= (1.0 + r_t)
        navs[rb_next] = nav * (1.0 - C.COST)
    return pd.Series(navs).sort_index()


def yearly_table(nav):
    """按月 nav 拆分成年度收益序列, 返回 DataFrame(year, ann, mdd, calmar, n)"""
    rets = nav / nav.shift(1) - 1
    rets = rets.dropna()
    rows = []
    for yr, grp in rets.groupby(lambda x: x[:4]):
        yr_nav = (1 + grp).cumprod()
        total = yr_nav.iloc[-1] - 1
        n = len(grp)
        ann = (1 + total) ** (12.0 / n) - 1 if n > 0 else 0.0
        hwm = yr_nav.cummax()
        mdd = (yr_nav / hwm - 1.0).min() if len(yr_nav) else 0.0
        calmar = ann / abs(mdd) if mdd < 0 else np.nan
        rows.append({"year": yr, "ann": ann, "mdd": mdd, "calmar": calmar, "n": n})
    return pd.DataFrame(rows).set_index("year")


def main():
    env = C.Env()
    out = []
    out.append("=" * 88)
    out.append("方向2 补充验证: HRP 参数敏感性 (窗口 60/120/180) + 分年 OOS")
    out.append("基准: 等权+MA20(0.98) ｜ 全区间 2020-2026, 月度调仓 20bps, RS12 弱段持 512100")
    out.append("")

    # ---- 1. 参数敏感性 ----
    out.append("【1】窗口敏感性 (全部叠加 MA20 三档 0.98)")
    txt, _ = C.metrics_table([
        ("等权+MA20", run_hrp(env, False, True, 120), {}),
        ("HRP(60)+MA20", run_hrp(env, True, True, 60), {}),
        ("HRP(120)+MA20", run_hrp(env, True, True, 120), {}),
        ("HRP(180)+MA20", run_hrp(env, True, True, 180), {}),
    ])
    out.append(txt)
    out.append("")

    # ---- 2. 分年 OOS ----
    out.append("【2】分年 OOS: 等权+MA20 vs HRP(120)+MA20")
    eq = run_hrp(env, False, True, 120)
    hrp = run_hrp(env, True, True, 120)
    t_eq = yearly_table(eq)
    t_hrp = yearly_table(hrp)
    out.append(f"{'年份':<6}{'等权年化':>10}{'HRP年化':>10}{'差pp':>8}{'等权MaxDD':>11}{'HRP MaxDD':>11}{'等权卡玛':>9}{'HRP卡玛':>9}")
    better = 0
    total_n = 0
    for yr in sorted(set(t_eq.index) | set(t_hrp.index)):
        a = t_eq.loc[yr] if yr in t_eq.index else None
        b = t_hrp.loc[yr] if yr in t_hrp.index else None
        if a is None or b is None:
            continue
        diff = (b["ann"] - a["ann"]) * 100
        flag = " *" if b["calmar"] > a["calmar"] else ""
        if b["calmar"] > a["calmar"]:
            better += 1
        total_n += 1
        out.append(f"{yr:<6}{a['ann']*100:>9.2f}%{b['ann']*100:>9.2f}%{diff:>7.2f}"
                   f"{a['mdd']*100:>10.2f}%{b['mdd']*100:>10.2f}%"
                   f"{a['calmar']:>9.2f}{b['calmar']:>9.2f}{flag}")
    out.append("")
    out.append(f"HRP 卡玛优于等权的年份: {better}/{total_n}")
    out.append("=" * 88)
    report = "\n".join(out)
    with open(os.path.join(C.OUT_DIR, "direction2_hrp_sensitivity.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
