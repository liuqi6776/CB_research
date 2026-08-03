# -*- coding: utf-8 -*-
"""方向B: 宏观 PMI 状态择时 (iFinD EDB 数据)
- 信号: 制造业PMI, 调仓月取"上月已发布值"(shift(1), 无前视), 三种弱态定义:
  - pmi49  : 上月 PMI < 49 (极端收缩)
  - pmi50x2: 连续 2 个月 PMI < 50 (弱于荣枯线)
  - pmi3ma : PMI 3月移动平均 < 50 (趋势弱)
- 叠加: 弱态时 MA20 仓位 ×0.5; 与方向1教训对比 (覆盖率是否可控)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.studies.study_008_enhancements import common as C

MA20_DEEP = 0.98


def load_pmi():
    df = pd.read_csv(os.path.join(C.DATA_DIR, "pmi_monthly.csv"), encoding="utf-8")
    df.columns = ["month", "pmi"]
    df["month"] = df["month"].astype(str).str.replace("-", "", regex=False)
    s = df.set_index("month")["pmi"].astype(float)
    return s


def pmi_weak(rb, pmi, mode):
    """rb: 调仓日 YYYYMMDD; pmi: 月序 Series(含当月); 返回该调仓月是否 PMI 弱态 (用上月已发布值)"""
    m = rb[:6]
    idx = list(pmi.index)
    if m not in idx:
        return False
    pos = idx.index(m)
    # 上月值
    if pos < 1:
        return False
    p1 = pmi.iloc[pos - 1]
    if mode == "pmi49":
        return p1 < 49.0
    if mode == "pmi50x2":
        if pos < 2:
            return p1 < 50.0
        p2 = pmi.iloc[pos - 2]
        return p1 < 50.0 and p2 < 50.0
    if mode == "pmi3ma":
        vals = pmi.iloc[max(0, pos - 3):pos]
        return len(vals) >= 2 and vals.mean() < 50.0
    return False


def run_bt(env, pmi, mode, use_ma20=True, halve=False):
    navs = {}
    weak_days = 0
    total_days = 0
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        nav = navs.get(rb, 1.0)
        if picks is None:
            navs[rb_next] = nav
            continue
        w = pd.Series(1.0 / len(picks), index=picks)
        cr = (comb * w.reindex(comb.columns)).sum(axis=1, min_count=1)
        pmi_off = pmi_weak(rb, pmi, mode) if halve else False
        for t in hold:
            total_days += 1
            r_t = e_ret.loc[t]
            if rs12_on:
                ww = 1.0
                if use_ma20:
                    c = env.idx_close_1.get(t, np.nan)
                    m = env.ma20_1.get(t, np.nan)
                    if np.isfinite(c) and np.isfinite(m):
                        ww = 1.0 if c >= m else (0.5 if c >= MA20_DEEP * m else 0.0)
                if pmi_off:
                    ww *= 0.5
                    weak_days += 1
                r_t = ww * cr.loc[t]
            nav *= (1.0 + r_t)
        navs[rb_next] = nav * (1.0 - C.COST)
    return pd.Series(navs).sort_index(), weak_days, total_days


def main():
    env = C.Env()
    pmi = load_pmi()
    out = []
    out.append("=" * 88)
    out.append("方向B: 宏观 PMI 状态择时 (iFinD EDB)")
    out.append("信号: 调仓月取上月已发布 PMI (shift(1) 无前视); 弱态时 MA20 仓位×0.5")
    out.append("")

    # 覆盖率
    coverage = {}
    for mode in ["pmi49", "pmi50x2", "pmi3ma"]:
        n = sum(1 for rb in env.rebal if pmi_weak(rb, pmi, mode))
        coverage[mode] = n / max(len(env.rebal), 1)
    out.append("调仓月弱态覆盖率: " + " | ".join(f"{k}={coverage[k]:.1%}" for k in coverage))
    out.append("")

    rows = []
    extras = {}
    nav_base, _, _ = run_bt(env, pmi, "pmi49", use_ma20=True, halve=False)
    rows.append(("+MA20(对照)", nav_base, {}))
    for mode in ["pmi49", "pmi50x2", "pmi3ma"]:
        nav, wd, td = run_bt(env, pmi, mode, use_ma20=True, halve=True)
        rows.append((f"+MA20+PMI({mode})", nav, {"弱态天数": f"{wd}({wd/max(td,1):.1%})"}))
    txt, _ = C.metrics_table(rows)
    out.append(txt)
    out.append("=" * 88)
    report = "\n".join(out)
    with open(os.path.join(C.OUT_DIR, "direction5_pmi.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    fig, ax = plt.subplots(figsize=(13, 6))
    for lb, nav, _ in rows:
        ax.plot(np.arange(len(nav)), nav.values, label=lb, lw=1.6)
    ax.set_title("宏观 PMI 择时增量 (BASE+VAL+MA20, 2020-2026)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "direction5_pmi.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
