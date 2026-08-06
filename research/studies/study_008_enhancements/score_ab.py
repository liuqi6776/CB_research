# -*- coding: utf-8 -*-
"""打分层 A/B (3.23): zscore 均值 vs 行业中性化 vs 分位数

在冻结基线 engine 上注入三种打分 (score_layer.make_score_fn), 对比:
  [1] 打分 IC (全体成分股截面 Spearman vs 持有期收益, IC/ICIR)
  [2] Top60 打分-收益散点 Spearman (3.18 口径)
  [3] 阈值选股回测 (等权, 阶段4, 万1): 基线 zscore >=0.93 对照,
      行业中性化/分位数各自重标定阈值后按"持仓对齐"档对比

输出: results/score_ab.txt|json + score_ab_ic.png + score_ab_scatter.png + score_ab_nav.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements import score_layer as SL
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df

THR_Z = 0.93                      # 基线 zscore 阈值 (3.19 选定)
SCAN_IN = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]   # ind_neut 阈值候选
SCAN_QT = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85]  # quantile 阈值候选 (分位排名)


def _metrics(s):
    if s is None or len(s) < 2:
        return None
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
    ind_map = C.load_industry_map()
    fns = {m: SL.make_score_fn(m, ind_map=ind_map) for m in SL.MODES}

    # 行业映射覆盖率 (成分股)
    cov_n, cov_d = [], []
    for rb in rebal:
        members = __import__("research.factor_dic.run_validation", fromlist=["load_index_weight"]).load_index_weight(rb)
        if not members:
            continue
        m = [c for c in members if c in ind_map]
        cov_n.append(len(m) / len(members))
    lines = ["打分层 A/B (3.23): zscore 均值 vs 行业中性化 vs 分位数", "=" * 92]
    lines.append(f"行业映射: {len(ind_map)} 只, 成分股行业覆盖率均值 {np.mean(cov_n):.1%}")

    # ---- [1] 打分 IC ----
    ics = {m: [] for m in SL.MODES}
    for rb in rebal:
        fr = _fwd_at(env, rb)
        for m in SL.MODES:
            scored = fns[m](env, rb)
            if scored is None:
                continue
            df = pd.DataFrame({"s": scored, "f": fr}).dropna()
            if len(df) > 30:
                rho, _ = spearmanr(df["s"], df["f"])
                ics[m].append(rho)

    def _ic_stat(a):
        a = np.asarray(a)
        return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                    icir=float(a.mean() / (a.std() + 1e-12) * np.sqrt(12.0)),
                    pos=float((a > 0).mean()))

    st = {m: _ic_stat(v) for m, v in ics.items()}
    lines.append("")
    lines.append("[1] 打分 IC (全体成分股截面 Spearman vs 持有期收益):")
    for m in SL.MODES:
        lines.append(f"    {m:<10} IC {st[m]['mean']:+.4f} | ICIR {st[m]['icir']:+.3f} | 正占比 {st[m]['pos']:.0%} | n={st[m]['n']}")
    lines.append(f"    行业中性化 - 基线: IC {st['ind_neut']['mean']-st['zscore']['mean']:+.4f} | "
                 f"ICIR {st['ind_neut']['icir']-st['zscore']['icir']:+.3f}")
    lines.append(f"    分位数     - 基线: IC {st['quantile']['mean']-st['zscore']['mean']:+.4f} | "
                 f"ICIR {st['quantile']['icir']-st['zscore']['icir']:+.3f}")

    # ---- [2] Top60 散点 ----
    pts = {m: [] for m in SL.MODES}
    for rb in rebal:
        fr = _fwd_at(env, rb)
        for m in SL.MODES:
            scored = fns[m](env, rb)
            if scored is None or len(scored) < 60:
                continue
            top = scored.sort_values(ascending=False).head(60).index
            df = pd.DataFrame({"s": scored.reindex(top), "f": fr.reindex(top)}).dropna()
            for code, row in df.iterrows():
                pts[m].append((rb, code, row["s"], row["f"]))
    d = {m: pd.DataFrame(v, columns=["rb", "code", "s", "f"]) for m, v in pts.items()}
    rho = {m: spearmanr(d[m]["s"], d[m]["f"])[0] for m in SL.MODES}
    lines.append("")
    lines.append("[2] Top60 打分-持有期收益散点 (3.18 口径):")
    for m in SL.MODES:
        lines.append(f"    {m:<10} n={len(d[m])} | Spearman {rho[m]:+.4f}")
    lines.append(f"    行业中性化 - 基线: {rho['ind_neut']-rho['zscore']:+.4f} | "
                 f"分位数 - 基线: {rho['quantile']-rho['zscore']:+.4f}")

    # ---- [3] 阈值选股回测 ----
    lines.append("")
    lines.append("[3] 阈值选股回测 (等权, 阶段4, 万1; 各模式独立重标定阈值):")
    bt = {}
    s_base, st_base = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                     e_ovn, e_intra, use_hrp=False, use_ma20=False,
                                     st_map=st_map, limit_sets=(one_up, one_dn),
                                     tradable=tf5, score_thr=THR_Z,
                                     score_fn=fns["zscore"])
    m_base = _metrics(s_base)
    ns0 = float(np.mean(st_base["n_selected"])) if st_base["n_selected"] else 0.0
    bt["zscore"] = dict(thr=THR_Z, m=m_base, ns=ns0)
    lines.append(f"    基线 zscore >= {THR_Z} (持仓 {ns0:.1f}): 终值 {m_base['final']:.4f} | "
                 f"年化 {m_base['ann']:.2%} | Sharpe {m_base['sharpe']:.2f} | "
                 f"MaxDD {m_base['mdd']:.2%} | 卡玛 {m_base['calmar']:.2f}")

    for mode, thrs in (("ind_neut", SCAN_IN), ("quantile", SCAN_QT)):
        scan = []
        for thr in thrs:
            s, stt = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                    e_ovn, e_intra, use_hrp=False, use_ma20=False,
                                    st_map=st_map, limit_sets=(one_up, one_dn),
                                    tradable=tf5, score_thr=thr, score_fn=fns[mode])
            m = _metrics(s)
            ns = float(np.mean(stt["n_selected"])) if stt["n_selected"] else 0.0
            if m is None:
                lines.append(f"    {mode:<10} >= {thr:<5}: 空净值 (无月份满足阈值), 跳过")
                continue
            scan.append((thr, m, ns))
            lines.append(f"    {mode:<10} >= {thr:<5}: 终值 {m['final']:.4f} | 年化 {m['ann']:.2%} | "
                         f"Sharpe {m['sharpe']:.2f} | MaxDD {m['mdd']:.2%} | 卡玛 {m['calmar']:.2f} | 持仓 {ns:.1f}")
        cand = sorted(scan, key=lambda t: abs(t[2] - ns0))
        if cand:
            thr_b, m_b, ns_b = cand[0]
            bt[mode] = dict(thr=thr_b, m=m_b, ns=ns_b)
            lines.append(f"    -> {mode} 持仓对齐档 >= {thr_b} (持仓 {ns_b:.1f} vs 基线 {ns0:.1f}): "
                         f"年化 {m_b['ann']:.2%} vs 基线 {m_base['ann']:.2%} | "
                         f"Sharpe {m_b['sharpe']:.2f} vs {m_base['sharpe']:.2f} | "
                         f"卡玛 {m_b['calmar']:.2f} vs {m_base['calmar']:.2f}")
    print("\n".join(lines))

    # ---- 图1: IC 时间序列 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = {"zscore": "#333", "ind_neut": "#1f77b4", "quantile": "#d62728"}
    ax = axes[0]
    for m in SL.MODES:
        ax.plot(range(len(ics[m])), ics[m], lw=1.0, alpha=0.7, color=colors[m], label=f"{m} (IC {st[m]['mean']:+.4f})")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("月度打分 IC: zscore vs 行业中性化 vs 分位数")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for m in SL.MODES:
        rol = pd.Series(ics[m]).rolling(12, min_periods=6).mean()
        ax.plot(rol.index, rol.values, lw=1.6, color=colors[m], label=f"{m} 12月滚动 IC")
    ax.axhline(0, color="#888", lw=0.8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "score_ab_ic.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    # ---- 图2: Top60 散点 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, m in zip(axes, SL.MODES):
        ax.scatter(d[m]["s"], d[m]["f"] * 100, s=6, alpha=0.35, color=colors[m])
        bins = pd.qcut(d[m]["s"].rank(method="first"), 10, labels=False)
        xm = d[m].groupby(bins)["s"].mean()
        ym = d[m].groupby(bins)["f"].mean() * 100
        ax.plot(xm.values, ym.values, "-o", color="orange", lw=1.5, ms=4)
        ax.set_title(f"{m}: Top60 n={len(d[m])}, Spearman {rho[m]:+.4f}")
        ax.set_xlabel("打分")
        ax.set_ylabel("持有期收益 %")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "score_ab_scatter.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    # ---- 图3: 回测净值对比 (各模式持仓对齐档) ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    navs = {"zscore": s_base}
    for mode in ("ind_neut", "quantile"):
        if mode in bt:
            s, _ = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                  e_ovn, e_intra, use_hrp=False, use_ma20=False,
                                  st_map=st_map, limit_sets=(one_up, one_dn),
                                  tradable=tf5, score_thr=bt[mode]["thr"], score_fn=fns[mode])
            navs[mode] = s
    for m, s in navs.items():
        ax.plot(s.index, s.values, lw=1.2, color=colors[m], label=f"{m} (>= {bt[m]['thr']}, 卡玛 {bt[m]['m']['calmar']:.2f})")
    ax.set_title("阈值选股净值对比 (等权, 阶段4, 万1; 各模式持仓对齐)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "score_ab_nav.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    out = dict(ind_cov=float(np.mean(cov_n)),
               ic={m: st[m] for m in SL.MODES},
               scatter={m: dict(n=len(d[m]), sp=float(rho[m])) for m in SL.MODES},
               bt={m: dict(thr=bt[m]["thr"], m=bt[m]["m"], n_sel=bt[m]["ns"]) for m in bt})
    with open(os.path.join(C.OUT_DIR, "score_ab.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(C.OUT_DIR, "score_ab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'score_ab.txt')} | .json")


if __name__ == "__main__":
    main()
