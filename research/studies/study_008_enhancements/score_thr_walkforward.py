# -*- coding: utf-8 -*-
"""≥0.93 分数阈值选股 — 多段 walk-forward 复核 (2026-08-05)

回答: 0.93 阈值是否为样本内过拟合? 在滚动 OOS 段是否稳定占优 v1.1.0 基线?

口径:
- 变体: A v1.1.0 基线 (Top60+IVW120+阶段4/5, 万1) / B >=thr 等权 (阶段4, 万1)
- [1] 全段连续回测 -> 按年切片 (生产连续口径, 与 risk_control_topn_walkforward 一致)
- [2] 4 段滚动定参: 定参窗 (不含 OOS 年) 内扫描阈值选最优 (卡玛优先) -> 在 OOS 年
      与固定 0.93、基线三方对比 (定参窗与 OOS 年日历对齐, 4 段)
输出: results/score_thr_walkforward.txt|json + score_thr_walkforward_oos.png
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.concentration import apply_concentration, amount60_at
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df

SCAN_THRS = [0.85, 0.89, 0.91, 0.93, 0.95, 0.97]
FIX_THR = 0.93
# 定参窗 (前 3 年, 不含 OOS 年) -> OOS 年 (与 risk_control_topn_walkforward 的 4 段日历对齐)
WF_WINS = [
    ("2023", "20200101", "20221231"),
    ("2024", "20210101", "20231231"),
    ("2025", "20220101", "20241231"),
    ("2026", "20230101", "20251231"),
]
OOS_WIN_END = "20261231"


def _metrics(s):
    n = len(s)
    if n < 30:
        return dict(final=float(s.iloc[-1]), ann=float("nan"), sharpe=float("nan"),
                    mdd=0.0, calmar=float("nan"), m_mean=float("nan"), m_win=float("nan"))
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
    ret = s.pct_change().dropna()
    shp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(242.0)
    dd = ((s.cummax() - s) / s.cummax()).max()
    m_ret = s.groupby(s.index.str[:6]).last().pct_change().dropna()
    return dict(final=float(s.iloc[-1]), ann=cagr, sharpe=shp, mdd=float(dd),
                calmar=float(cagr / dd) if dd > 0 else 0.0,
                m_mean=float(m_ret.mean()), m_win=float((m_ret > 0).mean()))


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)
    st_map = E.load_st_intervals()
    one_up, one_dn = E.build_limit_sets(open_df, high_df, low_df, pct_df, env.all_codes)
    amount_df = load_amount_df(env, td)
    ind_map = C.load_industry_map()
    tf5 = Tradability(td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                      st_map=st_map, min_vol=12.0, pct_df=pct_df)

    def _conc(rb, w, nav_pre):
        return apply_concentration(w, ind_map=ind_map, cap_stock=0.04, cap_ind=0.20,
                                   cap_top5=0.20,
                                   amount60=amount60_at(amount_df, td, rb),
                                   nav_pre=nav_pre, cap_amount=0.05, scale=1e8)

    def _run(thr=None, hrp=False, conc=False, w0=None, w1=None):
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=hrp, use_ma20=False,
                               st_map=st_map, limit_sets=(one_up, one_dn),
                               tradable=tf5, concentration=_conc if conc else None,
                               score_thr=None if hrp else thr, start=w0, end=w1)
        return s, st

    lines = ["分数阈值 ≥0.93 选股 — 多段 walk-forward 复核 (vs v1.1.0 基线, 万1)", "=" * 96]

    # ---- [1] 全段连续: 固定 0.93 vs 基线 ----
    s_base, _ = _run(hrp=True, conc=True)
    s_fix, st_fix = _run(thr=FIX_THR)
    m_base, m_fix = _metrics(s_base), _metrics(s_fix)
    lines.append("[1] 全段连续 (2020-01~2026-07, 生产口径):")
    lines.append(f"    v1.1.0 基线 (Top60+IVW120+集中度): 终值 {m_base['final']:.4f} | "
                 f"年化 {m_base['ann']:.2%} | Sharpe {m_base['sharpe']:.2f} | "
                 f"MaxDD {m_base['mdd']:.2%} | 卡玛 {m_base['calmar']:.2f}")
    lines.append(f"    >=0.93 等权 (固定阈值):           终值 {m_fix['final']:.4f} | "
                 f"年化 {m_fix['ann']:.2%} | Sharpe {m_fix['sharpe']:.2f} | "
                 f"MaxDD {m_fix['mdd']:.2%} | 卡玛 {m_fix['calmar']:.2f}")
    lines.append(f"    (持仓均值 {np.mean(st_fix['n_selected']):.1f} 只, "
                 f"fail-closed {st_fix['n_missing']} 月)")

    # ---- [2] 全段切片: 逐年 OOS (0.93 vs 基线) ----
    lines.append("")
    lines.append("[2] 全段连续回测后按年切片 (0.93 vs 基线):")
    lines.append(f"    {'年份':<6}{'基线年化':>9}{'0.93年化':>9}{'差pp':>8}"
                 f"{'基线Sharpe':>10}{'0.93Sharpe':>11}{'基线MaxDD':>10}{'0.93MaxDD':>10}")
    yearly = []
    for label, w0, w1 in WF_WINS:
        seg_b = s_base[(s_base.index >= w0) & (s_base.index <= w1)]
        seg_f = s_fix[(s_fix.index >= w0) & (s_fix.index <= w1)]
        mb, mf = _metrics(seg_b), _metrics(seg_f)
        yearly.append((label, mb, mf))
        lines.append(f"    {label:<6}{mb['ann']:>9.2%}{mf['ann']:>9.2%}{mf['ann']-mb['ann']:>+8.2%}"
                     f"{mb['sharpe']:>10.2f}{mf['sharpe']:>11.2f}"
                     f"{mb['mdd']:>10.2%}{mf['mdd']:>10.2%}")

    # ---- [3] 滚动定参: 窗内扫描选阈值 -> OOS 年验证 ----
    lines.append("")
    lines.append("[3] 滚动定参 (定参窗不含 OOS 年, 窗内扫描阈值按卡玛选最优):")
    wf_rows = []
    for label, tr0, tr1 in WF_WINS:
        # 定参窗内扫描
        cand = []
        for thr in SCAN_THRS:
            s, st = _run(thr=thr, w0=tr0, w1=tr1)
            m = _metrics(s)
            cand.append((thr, m))
        cand.sort(key=lambda t: (-t[1]["calmar"], -t[1]["ann"]))
        best_thr, best_m = cand[0]
        # OOS 年: 窗选阈值 vs 固定 0.93 vs 基线
        oos0, oos1 = label + "0101", OOS_WIN_END
        s_w, _ = _run(thr=best_thr, w0=oos0, w1=oos1)
        s_f, _ = _run(thr=FIX_THR, w0=oos0, w1=oos1)
        s_b, _ = _run(hrp=True, conc=True, w0=oos0, w1=oos1)
        m_w, m_f, m_b = _metrics(s_w), _metrics(s_f), _metrics(s_b)
        wf_rows.append(dict(oos=label, tr=(tr0, tr1), best_thr=best_thr,
                            tr_cal=best_m["calmar"], tr_ann=best_m["ann"],
                            w=dict(cagr=m_w["ann"], shp=m_w["sharpe"], dd=m_w["mdd"]),
                            fix=dict(cagr=m_f["ann"], shp=m_f["sharpe"], dd=m_f["mdd"]),
                            base=dict(cagr=m_b["ann"], shp=m_b["sharpe"], dd=m_b["mdd"])))
        lines.append(f"    定参窗 {tr0[:4]}-{tr1[:4]} -> OOS {label}: 窗内最优阈值 {best_thr} "
                     f"(卡玛 {best_m['calmar']:.2f}, 年化 {best_m['ann']:.2%})")
        lines.append(f"      OOS 年: 窗选>= {best_thr}  年化 {m_w['ann']:>8.2%} "
                     f"Sharpe {m_w['sharpe']:>5.2f} MaxDD {m_w['mdd']:>8.2%}")
        lines.append(f"              固定>=0.93 年化 {m_f['ann']:>8.2%} "
                     f"Sharpe {m_f['sharpe']:>5.2f} MaxDD {m_f['mdd']:>8.2%}")
        lines.append(f"              基线        年化 {m_b['ann']:>8.2%} "
                     f"Sharpe {m_b['sharpe']:>5.2f} MaxDD {m_b['mdd']:>8.2%}")
        print(f"[wf] OOS {label}: 窗选 {best_thr} | 0.93 {m_f['ann']:.2%} | 基线 {m_b['ann']:.2%}", flush=True)

    # ---- 结论 ----
    lines.append("")
    wf_win = sum(1 for r in wf_rows if r["fix"]["cagr"] > r["base"]["cagr"])
    thrs_ok = sum(1 for r in wf_rows if 0.90 <= r["best_thr"] <= 0.95)
    lines.append(f"结论: 固定 0.93 在 OOS 年占优基线 {wf_win}/4; 滚动定参最优阈值落在 "
                 f"0.90~0.95 区间 {thrs_ok}/4 段.")
    if wf_win >= 3 and thrs_ok >= 3:
        lines.append("判定: 0.93 阈值稳健, 非单点过拟合 — 建议通过后再切换每日服务.")
    else:
        lines.append("判定: 0.93 在部分 OOS 段不占优, 需谨慎 — 建议观察/改用动态阈值.")

    print("\n".join(lines))

    # ---- 图: OOS 年化对比 (3 组 x 4 段) ----
    labels = [r["oos"] for r in wf_rows]
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.5))
    vals = [r["w"]["cagr"] for r in wf_rows]
    ax.bar(x - w, vals, w, label="窗选最优阈值", color="#1f77b4", alpha=0.85)
    vals = [r["fix"]["cagr"] for r in wf_rows]
    ax.bar(x, vals, w, label="固定 ≥0.93", color="#d62728", alpha=0.85)
    vals = [r["base"]["cagr"] for r in wf_rows]
    ax.bar(x + w, vals, w, label="v1.1.0 基线", color="#333", alpha=0.85)
    for i, r in enumerate(wf_rows):
        ax.text(i - w, r["w"]["cagr"], f"{r['w']['cagr']:.1%}", ha="center", va="bottom", fontsize=8)
        ax.text(i, r["fix"]["cagr"], f"{r['fix']['cagr']:.1%}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w, r["base"]["cagr"], f"{r['base']['cagr']:.1%}", ha="center", va="bottom", fontsize=8)
        ax.text(i, -0.20, f"窗选≥{r['best_thr']}", ha="center", fontsize=9, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"OOS {l}" for l in labels])
    ax.set_title("分数阈值选股 walk-forward: OOS 年化对比 (滚动定参 vs 固定 0.93 vs 基线)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "score_thr_walkforward_oos.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    out = dict(full=dict(base=m_base, fix=m_fix, n_sel_mean=float(np.mean(st_fix["n_selected"]))),
               yearly=yearly, wf=wf_rows)
    with open(os.path.join(C.OUT_DIR, "score_thr_walkforward.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(C.OUT_DIR, "score_thr_walkforward.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'score_thr_walkforward.txt')} | .json")


if __name__ == "__main__":
    main()
