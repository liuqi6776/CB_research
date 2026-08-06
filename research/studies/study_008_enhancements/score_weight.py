# -*- coding: utf-8 -*-
"""因子加权实验 (3.25): 等权 zscore 均值 vs IC 加权 vs ICIR 加权

3.21/3.22 两次补因子实验均因"等权均值稀释强信号"失败; 本节不动因子集合 (BASE+VAL 4 因子),
只改合成权重, 直接验证加权能否提升打分区分度:

  等权(ew)   : w_f = 1/n                (基线, 与 engine.score_at 完全一致)
  IC加权(ic) : w_f = max(0, 扩展窗 IC 均值)   (弱因子自动降权, 负 IC 因子清零)
  ICIR加权   : w_f = max(0, 扩展窗 IC 均值/IC 标准差)  (加入稳定性惩罚)

无前视设计: 第 t 月权重只用 1..t-1 月的 IC (扩展窗), warmup 前 6 个月回落等权;
VAL 缺失月份自动在可用因子内重归一化。

流程: [1] 4 因子月度 IC 表 → [2] 权重轨迹 → [3] 打分 IC/Top60 散点 →
      [4] 各方案阈值重标定回测 (持仓对齐 ~54.5) vs 基线 >=0.93

输出: results/score_weight.txt|json + score_weight_ic.png + score_weight_w.png + score_weight_nav.png
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
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df
from research.factor_dic import run_validation as rv
from research.factor_dic import style_factors as sf

FACTORS = ["ret_1m", "ivol", "turn", "VAL"]
THR_EW = 0.93
SCAN = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
WARMUP = 6


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


def _zdf(fdf):
    return fdf.apply(sf.winsorize_series).apply(
        lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)


def monthly_ic_tab(env, rebal):
    """DataFrame(rb × factor): 截面 Spearman(winsorize+zscore 因子, 持有期收益)"""
    rows = []
    for rb in rebal:
        fdf = E.build_fdf(env, rb, None)
        if fdf is None:
            continue
        z = _zdf(fdf)
        fr = _fwd_at(env, rb)
        row = {"rb": rb}
        for f in FACTORS:
            if f not in z.columns:
                continue
            df = pd.DataFrame({"z": z[f], "f": fr}).dropna()
            if len(df) > 30:
                rho, _ = spearmanr(df["z"], df["f"])
                row[f] = float(rho)
        rows.append(row)
    return pd.DataFrame(rows).set_index("rb")


def weights_series(ic_tab, scheme):
    """返回 {rb: Series(因子权重)}; warmup 前或全负 IC 时回落等权"""
    out = {}
    for i, rb in enumerate(ic_tab.index):
        if scheme == "ew" or i < WARMUP:
            out[rb] = pd.Series(1.0 / len(FACTORS), index=FACTORS)
            continue
        hist = ic_tab.iloc[:i][FACTORS]
        if scheme == "ic":
            w = hist.mean().clip(lower=0.0)
        else:  # icir
            w = (hist.mean() / (hist.std() + 1e-9)).clip(lower=0.0)
        if w.sum() <= 0:
            out[rb] = pd.Series(1.0 / len(FACTORS), index=FACTORS)
        else:
            out[rb] = w / w.sum()
    return out


def score_weighted(env, rb, weights):
    """等权/加权合成打分 (无 VAL 月自动在可用因子内重归一化)"""
    fdf = E.build_fdf(env, rb, None)
    if fdf is None:
        return None
    z = _zdf(fdf)
    cols = [c for c in FACTORS if c in z.columns]
    has = z[cols].dropna()
    if len(has) < rv.TOP_N:
        return None
    if weights is None or weights.sum() <= 0:
        return has.mean(axis=1)
    w = weights.reindex(cols).fillna(0.0)
    if w.sum() <= 0:
        return has.mean(axis=1)
    w = w / w.sum()
    return has.dot(w)


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

    print("[w] 计算 4 因子月度 IC 表 ...", flush=True)
    ic_tab = monthly_ic_tab(env, rebal)
    wts = {s: weights_series(ic_tab, s) for s in ("ew", "ic", "icir")}

    lines = ["因子加权实验 (3.25): 等权 vs IC 加权 vs ICIR 加权 (BASE+VAL 4 因子, 扩展窗无前视)",
             "=" * 92]
    lines.append(f"月度 IC 表: {len(ic_tab)} 月; warmup {WARMUP} 月内回落等权; 合成含 VAL 月 {int(ic_tab['VAL'].notna().sum())} 个")
    lines.append("")
    lines.append("[1] 4 因子月度 IC 统计 (全期):")
    for f in FACTORS:
        s = ic_tab[f].dropna()
        if len(s):
            lines.append(f"    {f:<10} IC {s.mean():+.4f} | ICIR {s.mean()/(s.std()+1e-9):+.3f} | 正占比 {(s>0).mean():.0%} | n={len(s)}")

    # ---- [2] 权重轨迹 (每 6 月抽样) ----
    lines.append("")
    lines.append("[2] 加权方案权重轨迹 (warmup 后每 12 月抽样):")
    wt_lines = {}
    for s in ("ic", "icir"):
        rows = [f"    {s:<5} " + " ".join(f"{f}={wts[s][rb][f]:.2f}" for f in FACTORS)
                for rb in ic_tab.index[::12] if rb in wts[s]]
        wt_lines[s] = rows
        for r in rows:
            lines.append(r)

    # ---- [3] 打分 IC + Top60 散点 ----
    ics = {s: [] for s in ("ew", "ic", "icir")}
    pts = {s: [] for s in ("ew", "ic", "icir")}
    for rb in ic_tab.index:
        fr = _fwd_at(env, rb)
        for s in ("ew", "ic", "icir"):
            scored = score_weighted(env, rb, wts[s].get(rb))
            if scored is None:
                continue
            df = pd.DataFrame({"s": scored, "f": fr}).dropna()
            if len(df) > 30:
                rho, _ = spearmanr(df["s"], df["f"])
                ics[s].append(rho)
            if len(scored) >= 60:
                top = scored.sort_values(ascending=False).head(60).index
                df2 = pd.DataFrame({"s": scored.reindex(top), "f": fr.reindex(top)}).dropna()
                for code, row in df2.iterrows():
                    pts[s].append((rb, code, row["s"], row["f"]))
    lines.append("")
    lines.append("[3] 打分 IC / Top60 散点:")
    d = {}
    for s in ("ew", "ic", "icir"):
        a = np.asarray(ics[s])
        df = pd.DataFrame(pts[s], columns=["rb", "code", "s", "f"])
        d[s] = df
        rho = spearmanr(df["s"], df["f"])[0] if len(df) else np.nan
        lines.append(f"    {s:<6} IC {a.mean():+.4f} | ICIR {a.mean()/(a.std()+1e-12):+.3f} | 正占比 {(a>0).mean():.0%} "
                     f"| Top60 Spearman {rho:+.4f} (n={len(df)})")
    lines.append(f"    与等权差值: IC {np.mean(ics['ic'])-np.mean(ics['ew']):+.4f} (ic) | "
                 f"{np.mean(ics['icir'])-np.mean(ics['ew']):+.4f} (icir); "
                 f"Top60 {spearmanr(d['ic']['s'], d['ic']['f'])[0]-spearmanr(d['ew']['s'], d['ew']['f'])[0]:+.4f} (ic) | "
                 f"{spearmanr(d['icir']['s'], d['icir']['f'])[0]-spearmanr(d['ew']['s'], d['ew']['f'])[0]:+.4f} (icir)")

    # ---- [4] 阈值回测 ----
    lines.append("")
    lines.append("[4] 阈值选股回测 (等权, 阶段4, 万1; 各方案独立重标定, 持仓对齐):")
    fns = {s: (lambda env, rb, s=s: score_weighted(env, rb, wts[s].get(rb))) for s in ("ew", "ic", "icir")}
    bt = {}
    for s in ("ew", "ic", "icir"):
        scan = []
        for thr in (SCAN if s != "ew" else [THR_EW]):
            nav, stt = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                      e_ovn, e_intra, use_hrp=False, use_ma20=False,
                                      st_map=st_map, limit_sets=(one_up, one_dn),
                                      tradable=tf5, score_thr=thr, score_fn=fns[s])
            m = _metrics(nav)
            ns = float(np.mean(stt["n_selected"])) if stt["n_selected"] else 0.0
            if m is None:
                continue
            scan.append((thr, m, ns, nav))
            lines.append(f"    {s:<6} >= {thr:<4}: 终值 {m['final']:.4f} | 年化 {m['ann']:.2%} | "
                         f"Sharpe {m['sharpe']:.2f} | 卡玛 {m['calmar']:.2f} | 持仓 {ns:.1f}")
        if not scan:
            continue
        if s == "ew":
            bt[s] = dict(thr=THR_EW, m=scan[0][1], ns=scan[0][2], nav=scan[0][3])
            ns_ew = scan[0][2]
        else:
            cand = sorted(scan, key=lambda t: abs(t[2] - ns_ew))
            thr_b, m_b, ns_b, nav_b = cand[0]
            bt[s] = dict(thr=thr_b, m=m_b, ns=ns_b, nav=nav_b)
            lines.append(f"    -> {s} 持仓对齐档 >= {thr_b} (持仓 {ns_b:.1f} vs 等权 {ns_ew:.1f}): "
                         f"年化 {m_b['ann']:.2%} vs 等权 {bt['ew']['m']['ann']:.2%} | "
                         f"Sharpe {m_b['sharpe']:.2f} vs {bt['ew']['m']['sharpe']:.2f} | "
                         f"卡玛 {m_b['calmar']:.2f} vs {bt['ew']['m']['calmar']:.2f}")
    print("\n".join(lines))

    # ---- 图1: 月度 IC 对比 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = {"ew": "#333", "ic": "#1f77b4", "icir": "#d62728"}
    ax = axes[0]
    for s in ("ew", "ic", "icir"):
        ax.plot(range(len(ics[s])), ics[s], lw=1.0, alpha=0.65, color=colors[s],
                label=f"{s} (IC {np.mean(ics[s]):+.4f})")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("月度打分 IC: 等权 vs IC加权 vs ICIR加权")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for s in ("ew", "ic", "icir"):
        rol = pd.Series(ics[s]).rolling(12, min_periods=6).mean()
        ax.plot(rol.index, rol.values, lw=1.6, color=colors[s], label=f"{s} 12月滚动")
    ax.axhline(0, color="#888", lw=0.8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "score_weight_ic.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    # ---- 图2: 权重轨迹 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, s in zip(axes, ("ic", "icir")):
        W = pd.DataFrame(wts[s]).T.reindex(ic_tab.index).fillna(method="ffill")
        for f in FACTORS:
            ax.plot(W.index, W[f], lw=1.2, label=f)
        ax.set_title(f"因子权重轨迹 ({s}, 扩展窗, warmup={WARMUP})")
        ax.legend(loc="center right", fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "score_weight_w.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    # ---- 图3: 净值对比 ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for s in ("ew", "ic", "icir"):
        if s in bt:
            ax.plot(bt[s]["nav"].index, bt[s]["nav"].values, lw=1.2, color=colors[s],
                    label=f"{s} (>= {bt[s]['thr']}, 卡玛 {bt[s]['m']['calmar']:.2f})")
    ax.set_title("阈值选股净值对比 (持仓对齐)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "score_weight_nav.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    out = dict(ic_tab_mean={f: float(ic_tab[f].mean()) for f in FACTORS},
               ic={s: dict(n=len(ics[s]), mean=float(np.mean(ics[s])),
                           icir=float(np.mean(ics[s]) / (np.std(ics[s]) + 1e-12)),
                           pos=float((np.asarray(ics[s]) > 0).mean()),
                           top60_sp=float(spearmanr(d[s]["s"], d[s]["f"])[0])) for s in ("ew", "ic", "icir")},
               bt={s: dict(thr=bt[s]["thr"], m=bt[s]["m"], n_sel=bt[s]["ns"]) for s in bt})
    with open(os.path.join(C.OUT_DIR, "score_weight.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(C.OUT_DIR, "score_weight.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'score_weight.txt')} | .json")


if __name__ == "__main__":
    main()
