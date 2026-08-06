# -*- coding: utf-8 -*-
"""因子增强实验 v2: BASE+VAL(4因子) vs +8 低相关量价因子

流程:
  [0] 单因子方向检验 (winsorize+zscore 后与 fwd 持有收益的 IC), 负 IC 因子自动翻转
  [1] 增强打分 IC vs 基线 (全体成分股截面 Spearman)
  [2] Top60 打分-收益散点对比 (3.18 口径)
  [3] 增强打分阈值扫描 (持仓量与基线 0.93 档对齐) + 基线 0.93 对照
      注: 打分=zscore均值, 因子越多分布越窄 (std~1/sqrt(n)), 同一阈值语义不同,
          必须重标定阈值后才能公平对比.

输出: results/factor_extend.txt|json + factor_extend_ic.png + factor_extend_scatter.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements import factor_lib as FL
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df

THR = 0.93
SCAN_THRS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


def _metrics(s):
    n = len(s)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
    ret = s.pct_change().dropna()
    shp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(242.0)
    dd = ((s.cummax() - s) / s.cummax()).max()
    m_ret = s.groupby(s.index.str[:6]).last().pct_change().dropna()
    return dict(final=float(s.iloc[-1]), ann=cagr, sharpe=shp, mdd=float(dd),
                calmar=float(cagr / dd) if dd > 0 else 0.0,
                m_mean=float(m_ret.mean()), m_win=float((m_ret > 0).mean()))


def _fwd_at(env, rb):
    out = {}
    for code, fr in env.fwd.items():
        if rb in fr.index:
            v = fr.loc[rb]
            if np.isfinite(v):
                out[code] = float(v)
    return pd.Series(out)


def _factor_ic(ext, name, env, rebal):
    """单因子 IC 序列 (winsorize+zscore 后 vs fwd)"""
    ics = []
    for rb in rebal:
        p = ext[name].get(rb)
        fr = _fwd_at(env, rb)
        if p is None or fr.empty or len(p) < 30:
            continue
        s = sf.winsorize_series(p)
        s = (s - s.mean()) / (s.std() + 1e-8)
        df = pd.DataFrame({"s": s, "f": fr}).dropna()
        if len(df) > 30:
            rho, _ = spearmanr(df["s"], df["f"])
            ics.append(rho)
    if not ics:
        return 0.0, 0
    a = np.asarray(ics)
    return float(a.mean()), int(a.size)


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)
    rebal = [rb for rb, *_ in env.month_segments()]
    st_map = E.load_st_intervals()
    one_up, one_dn = E.build_limit_sets(open_df, high_df, low_df, pct_df, env.all_codes)
    amount_df = load_amount_df(env, td)
    tf5 = Tradability(td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                      st_map=st_map, min_vol=12.0, pct_df=pct_df)

    print("[factor] 计算 8 个增强因子面板 ...", flush=True)
    stocks, _, _, _, _ = __import__("research.factor_dic.run_validation", fromlist=["load_panels"]).load_panels(
        td, env.all_codes, None)
    ext = FL.build_ext_panels(stocks, rebal, env.all_codes)

    lines = ["因子增强实验 v2: BASE+VAL(4因子) vs +8 低相关量价因子", "=" * 92]
    lines.append("[0] 单因子方向检验 (IC 均值, winsorize+zscore vs 持有期收益):")
    flips = []
    for name, sign, _ in FL.FACTOR_DEFS:
        m, n = _factor_ic(ext, name, env, rebal)
        mark = ""
        if m < 0:
            flips.append(name)
            ext[name] = {rb: -s for rb, s in ext[name].items()}
            m2, _ = _factor_ic(ext, name, env, rebal)
            mark = f"  -> 翻转后 IC {m2:+.4f}"
            m = m2
        lines.append(f"    {name:<14} IC {m:+.4f} ({n} 月){mark}")
    lines.append(f"    翻转因子: {flips if flips else '无'}")

    # ---- [1] IC 对比 ----
    ics0, ics1 = [], []
    for rb in rebal:
        fr = _fwd_at(env, rb)
        s0 = E.score_at(env, rb, None)
        s1 = E.score_at(env, rb, ext)
        for scored, ics in ((s0, ics0), (s1, ics1)):
            if scored is None:
                continue
            df = pd.DataFrame({"s": scored, "f": fr}).dropna()
            if len(df) > 30:
                rho, _ = spearmanr(df["s"], df["f"])
                ics.append(rho)

    def _ic_stat(ics):
        a = np.asarray(ics)
        return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                    icir=float(a.mean() / (a.std() + 1e-12) * np.sqrt(12.0)),
                    pos=float((a > 0).mean()))

    st0, st1 = _ic_stat(ics0), _ic_stat(ics1)
    lines.append("")
    lines.append("[1] 打分 IC (全体成分股截面 Spearman vs 持有期收益):")
    lines.append(f"    基线 4因子: IC 均值 {st0['mean']:+.4f} | ICIR {st0['icir']:+.3f} | 正占比 {st0['pos']:.0%}")
    lines.append(f"    增强12因子: IC 均值 {st1['mean']:+.4f} | ICIR {st1['icir']:+.3f} | 正占比 {st1['pos']:.0%}")
    lines.append(f"    差值: IC {st1['mean']-st0['mean']:+.4f} | ICIR {st1['icir']-st0['icir']:+.3f}")

    # ---- [2] Top60 散点对比 ----
    pts0, pts1 = [], []
    for rb in rebal:
        fr = _fwd_at(env, rb)
        for scored, pts in ((E.score_at(env, rb, None), pts0), (E.score_at(env, rb, ext), pts1)):
            if scored is None or len(scored) < 60:
                continue
            top = scored.sort_values(ascending=False).head(60).index
            df = pd.DataFrame({"s": scored.reindex(top), "f": fr.reindex(top)}).dropna()
            for code, row in df.iterrows():
                pts.append((rb, code, row["s"], row["f"]))
    d0 = pd.DataFrame(pts0, columns=["rb", "code", "s", "f"])
    d1 = pd.DataFrame(pts1, columns=["rb", "code", "s", "f"])
    r0, p0 = spearmanr(d0["s"], d0["f"])
    r1, p1 = spearmanr(d1["s"], d1["f"])
    lines.append("")
    lines.append("[2] Top60 打分-持有期收益散点 (3.18 口径):")
    lines.append(f"    基线 4因子: n={len(d0)} | Spearman {r0:+.4f} (p={p0:.2e})")
    lines.append(f"    增强12因子: n={len(d1)} | Spearman {r1:+.4f} (p={p1:.2e})")

    # ---- [3] 增强打分阈值扫描 (持仓对齐) + 基线 0.93 对照 ----
    lines.append("")
    lines.append("[3] 阈值选股回测 (等权, 阶段4, 万1; 增强打分需重标定阈值):")
    s_base, st_base = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                     e_ovn, e_intra, use_hrp=False, use_ma20=False,
                                     st_map=st_map, limit_sets=(one_up, one_dn),
                                     tradable=tf5, score_thr=THR, ext_panels=None)
    m_base = _metrics(s_base)
    ns0 = np.mean(st_base["n_selected"])
    lines.append(f"    基线打分 >=0.93 (持仓 {ns0:.1f}): 终值 {m_base['final']:.4f} | "
                 f"年化 {m_base['ann']:.2%} | Sharpe {m_base['sharpe']:.2f} | "
                 f"MaxDD {m_base['mdd']:.2%} | 卡玛 {m_base['calmar']:.2f}")

    scan = []
    for thr in SCAN_THRS:
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=False, use_ma20=False,
                               st_map=st_map, limit_sets=(one_up, one_dn),
                               tradable=tf5, score_thr=thr, ext_panels=ext)
        m = _metrics(s)
        ns = float(np.mean(st["n_selected"])) if st["n_selected"] else 0.0
        scan.append((thr, m, ns))
        lines.append(f"    增强 >= {thr:<4}: 终值 {m['final']:.4f} | 年化 {m['ann']:.2%} | "
                     f"Sharpe {m['sharpe']:.2f} | MaxDD {m['mdd']:.2%} | 卡玛 {m['calmar']:.2f} | "
                     f"持仓 {ns:.1f}")
    # 找持仓与基线 0.93 档 (54.5) 最接近的增强档
    cand = sorted(scan, key=lambda t: abs(t[2] - ns0))
    if cand:
        thr_best, m_best, ns_best = cand[0]
        lines.append(f"    -> 持仓对齐档: 增强 >= {thr_best} (持仓 {ns_best:.1f} vs 基线 {ns0:.1f}): "
                     f"年化 {m_best['ann']:.2%} vs 基线 {m_base['ann']:.2%} | "
                     f"Sharpe {m_best['sharpe']:.2f} vs {m_base['sharpe']:.2f}")
    print("\n".join(lines))

    # ---- 图1: IC 时间序列 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax = axes[0]
    ax.plot(range(len(ics0)), ics0, lw=1.0, color="#333", alpha=0.7, label="基线 4因子")
    ax.plot(range(len(ics1)), ics1, lw=1.0, color="#d62728", alpha=0.7, label="增强 12因子")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axhline(st0["mean"], color="#333", ls="--", lw=1.2)
    ax.axhline(st1["mean"], color="#d62728", ls="--", lw=1.2)
    ax.set_title("月度打分 IC: 基线 vs 增强 (+8 量价因子, 方向修正后)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]
    rol0 = pd.Series(ics0).rolling(12, min_periods=6).mean()
    rol1 = pd.Series(ics1).rolling(12, min_periods=6).mean()
    ax.plot(rol0.index, rol0.values, lw=1.6, color="#333", label="基线 12月滚动 IC")
    ax.plot(rol1.index, rol1.values, lw=1.6, color="#d62728", label="增强 12月滚动 IC")
    ax.axhline(0, color="#888", lw=0.8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "factor_extend_ic.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    # ---- 图2: Top60 散点对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, d, r, lbl, c in ((axes[0], d0, r0, "基线 4因子", "#333"),
                             (axes[1], d1, r1, "增强 12因子", "#d62728")):
        ax.scatter(d["s"], d["f"] * 100, s=6, alpha=0.35, color=c)
        bins = pd.qcut(d["s"].rank(method="first"), 10, labels=False)
        g = d.groupby(bins)["f"]
        xm = d.groupby(bins)["s"].mean()
        ym = g.mean() * 100
        ax.plot(xm.values, ym.values, "-o", color="orange", lw=1.5, ms=4)
        ax.set_title(f"{lbl}: Top60 散点 n={len(d)}, Spearman {r:+.4f}")
        ax.set_xlabel("打分 (zscore 均值)")
        ax.set_ylabel("持有期收益 %")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "factor_extend_scatter.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    out = dict(flips=flips, ic=dict(base=st0, ext=st1),
               scatter=dict(base=dict(n=len(d0), sp=float(r0)), ext=dict(n=len(d1), sp=float(r1))),
               bt=dict(base=dict(m=m_base, n_sel=ns0),
                       ext_scan=[dict(thr=t, m=m, n_sel=ns) for t, m, ns in scan]))
    with open(os.path.join(C.OUT_DIR, "factor_extend.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(C.OUT_DIR, "factor_extend.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'factor_extend.txt')} | .json")


if __name__ == "__main__":
    main()
