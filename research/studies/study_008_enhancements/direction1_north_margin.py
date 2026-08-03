# -*- coding: utf-8 -*-
"""方向1: 北向资金 + 两融余额信号 叠加 MA20 三档风控
信号(全部 T-1 已知, T 日生效, 无前视):
  - 两融预警: 全市场融资余额 20日变化率 < 0  (SSE+SZSE+BSE 合并, 全区间 2020-2026)
  - 北向预警: 北向 20日累计净买额 < 0          (2020-01~2024-08 有效, 此后停披露)
叠加规则: 预警触发时 MA20 仓位 ×0.5
对比: 无风控 / MA20 / MA20+两融(全区间) / MA20+北向 / MA20+北向∪两融(子区间)
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


def build_signals(env):
    """返回 dict: name -> pd.Series(index=交易日, 0/1, T-1 信号)"""
    sig = {}
    # 两融: T-1 信号
    margin = C.load_margin_series(env)
    chg20 = margin / margin.shift(20) - 1.0
    sig["margin"] = (chg20 < 0).astype(float).shift(1).fillna(0.0)
    # 北向: T-1 信号
    north = C.load_north_series()
    cum20 = north.rolling(20).sum()
    sig["north"] = (cum20 < 0).astype(float).shift(1).fillna(0.0)
    return sig


def run(env, sig, use=("margin",), sub_until=None):
    """主循环: 无风控基线 + MA20 + MA20+预警变体"""
    navs = {"BASE": {}, "MA20": {}, "MA20+AL": {}}
    st = {}
    for lb in navs:
        st[lb] = 1.0
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if sub_until is not None and rb_next > sub_until:
            break
        for lb in navs:
            nav = navs[lb].get(rb, 1.0)
            if picks is None:
                navs[lb][rb_next] = nav
                continue
            cr = comb.mean(axis=1)
            for t in hold:
                r_t = e_ret.loc[t]
                if rs12_on:
                    w = 1.0
                    if lb in ("MA20", "MA20+AL"):
                        c = env.idx_close_1.get(t, np.nan)
                        m = env.ma20_1.get(t, np.nan)
                        if np.isfinite(c) and np.isfinite(m):
                            w = 1.0 if c >= m else (0.5 if c >= 0.98 * m else 0.0)
                    if lb == "MA20+AL":
                        al = max([sig[k].get(t, 0.0) for k in use])
                        w *= 0.5 if al > 0 else 1.0
                    r_t = w * cr.loc[t]
                nav *= (1.0 + r_t)
            navs[lb][rb_next] = nav * (1.0 - C.COST)
    return {lb: pd.Series(navs[lb]).sort_index() for lb in navs}


def main():
    env = C.Env()
    sig = build_signals(env)

    # ---- 全区间: 两融 ----
    navs_full = run(env, sig, use=("margin",))
    full_rows = [
        ("BASE+VAL 无风控", navs_full["BASE"], {}),
        ("+MA20三档(0.98)", navs_full["MA20"], {}),
        ("+MA20+两融预警×0.5", navs_full["MA20+AL"], {"信号": "融资余额20日降"}),
    ]
    full_txt, _ = C.metrics_table(full_rows)

    # ---- 北向子区间 2020-01~2024-08 ----
    sub = "20240816"
    base = run(env, sig, use=("margin",), sub_until=sub)
    north = run(env, sig, use=("north",), sub_until=sub)
    both = run(env, sig, use=("north", "margin"), sub_until=sub)
    sub_rows = [
        ("BASE+VAL 无风控", base["BASE"], {}),
        ("+MA20三档(0.98)", base["MA20"], {}),
        ("+MA20+两融预警", base["MA20+AL"], {}),
        ("+MA20+北向预警", north["MA20+AL"], {}),
        ("+MA20+北向∪两融", both["MA20+AL"], {}),
    ]
    sub_txt, _ = C.metrics_table(sub_rows)

    report = []
    report.append("=" * 80)
    report.append("方向1: 北向资金 + 两融余额信号 叠加 MA20 三档")
    report.append("信号: 两融=全市场融资余额20日变化率<0; 北向=北向20日累计净买<0; 均 T-1 生效, 预警时 MA20 仓位×0.5")
    report.append("")
    report.append("【全区间 2020-01~2026-06】")
    report.append(full_txt)
    report.append("")
    report.append(f"【北向子区间 2020-01~{sub[:4]}-{sub[4:6]}】(北向2024-08后停止披露, 故子区间对比)")
    report.append(sub_txt)

    # 预警覆盖率
    margin = C.load_margin_series(env)
    chg20 = margin / margin.shift(20) - 1.0
    al = (chg20 < 0).astype(float)
    report.append("")
    report.append(f"两融预警覆盖率(交易日): {al.mean()*100:.1f}%")
    north = C.load_north_series()
    cum20 = north.rolling(20).sum()
    aln = (cum20 < 0).astype(float)
    report.append(f"北向预警覆盖率(2020-01~2024-08): {aln.mean()*100:.1f}%")
    report.append("=" * 80)

    txt = "\n".join(report)
    with open(os.path.join(C.OUT_DIR, "direction1_north_margin.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, navs, title in [
        (axes[0], navs_full, "全区间: 两融预警"),
        (axes[1], base, "北向子区间: 北向/两融预警"),
    ]:
        for lb, nav in navs.items():
            ax.plot(np.arange(len(nav)), nav.values, label=lb, lw=1.6)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "direction1_north_margin.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
