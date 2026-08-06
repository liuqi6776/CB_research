# -*- coding: utf-8 -*-
"""数据端增量 (3.28): PIT 财务因子作为增量信息源 vs BASE+VAL 基线

3.27 已实证财务腿 PIT 干净 (0 前视 / 重述率 0.00% / 覆盖 99.7% / 月龄中位 3 月)。
本节验证"财务信息"是否真的是增量 (现有 4 因子 = ret_1m/ivol/turn/VAL 全为量价+估值,
ROE/GM/LEV/GROW/SGROW 来自 fina_indicator 报表, 是 VAL 未覆盖的信息维度):

  [0] 单因子 IC: 估值 4 面板 (EP/BP/SP/DP) + 财务 5 面板 (ROE/GM/LEV/GROW/SGROW),
      winsorize+zscore vs 持有期收益, 负 IC 因子翻转
  [1] 正交性: 财务因子 zscore 与现有 4 因子 (ret_1m/ivol/turn/VAL) 的截面相关
      (|corr|<0.3 = 正交 = 有增量信息; 另看财务因子内部相关 = 质量系/成长系冗余度)
  [2] 打分 IC: 基线 4因子 vs 增量组合 (全体成分股截面 Spearman)
  [3] 阈值选股回测 (等权+阶段4+万1, 各组合独立重标定阈值、持仓对齐基线 ~54.5)

组合 (财务因子均等权加入, 与基线 zscore 等权均值合成一致):
  base : BASE+VAL (4因子, 基线)
  q1   : +ROE (单因子质量)
  q3   : +ROE+GM+LEV (质量系)
  g2   : +GROW+SGROW (成长系)
  all  : +ROE+GM+LEV+GROW+SGROW (财务全量, 9因子)

输出: results/funda_factor.txt|json + funda_factor_ic.png + funda_factor_scatter.png
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
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df

THR = 0.93
FUNDA = ["ROE", "GM", "LEV", "GROW", "SGROW"]
VALU = ["EP", "BP", "SP", "DP"]
BASE_F = ["ret_1m", "ivol", "turn", "VAL"]
COMBOS = {
    "q1":  ["ROE"],
    "q3":  ["ROE", "GM", "LEV"],
    "g2":  ["GROW", "SGROW"],
    "all": FUNDA,
}


def _metrics(s):
    n = len(s)
    if n < 2:
        return dict(final=float(s.iloc[-1]) if len(s) else 1.0, ann=0.0, sharpe=0.0,
                    mdd=0.0, calmar=0.0, m_mean=0.0, m_win=0.0)
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


def _zs(s):
    s = sf.winsorize_series(s)
    return (s - s.mean()) / (s.std() + 1e-8)


def _ic_series(env, rebal, scored_fn):
    """打分/因子 vs fwd 的月度 Spearman IC 序列"""
    ics = []
    for rb in rebal:
        s = scored_fn(env, rb)
        fr = _fwd_at(env, rb)
        if s is None or fr.empty or len(s) < 30:
            continue
        df = pd.DataFrame({"s": s, "f": fr}).dropna()
        if len(df) > 30:
            rho, _ = spearmanr(df["s"], df["f"])
            ics.append(rho)
    return ics


def _ic_stat(ics):
    a = np.asarray(ics)
    if a.size == 0:
        return dict(n=0, mean=0.0, std=0.0, icir=0.0, pos=0.0)
    return dict(n=int(a.size), mean=float(a.mean()), std=float(a.std()),
                icir=float(a.mean() / (a.std() + 1e-12) * np.sqrt(12.0)),
                pos=float((a > 0).mean()))


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

    panels = env.panels          # common.Env 已构造: EP/BP/SP/DP/ROE/GM/LEV/GROW/SGROW + VAL
    lines = ["数据端增量 (3.28): PIT 财务因子 vs BASE+VAL 基线", "=" * 92]
    lines.append(f"财务面板: {FUNDA}, 估值面板: {VALU} (VAL=BP+SP+DP zscore 均值, 已含于基线)")

    # ---- [0] 单因子方向检验 + IC ----
    lines.append("")
    lines.append("[0] 单因子 IC (winsorize+zscore vs 持有期收益, 负 IC 自动翻转):")
    flips = []
    ic1 = {}
    for name in VALU + FUNDA:
        m, n = 0.0, 0
        ics = []
        for rb in rebal:
            p = panels[name].get(rb)
            fr = _fwd_at(env, rb)
            if p is None or fr.empty or len(p) < 30:
                continue
            s = _zs(p)
            df = pd.DataFrame({"s": s, "f": fr}).dropna()
            if len(df) > 30:
                rho, _ = spearmanr(df["s"], df["f"])
                ics.append(rho)
                n += 1
        if ics:
            m = float(np.mean(ics))
        mark = ""
        if m < 0:
            flips.append(name)
            panels[name] = {rb: -s for rb, s in panels[name].items()}
            # 翻转后重算 IC
            ics2 = []
            for rb in rebal:
                p = panels[name].get(rb)
                fr = _fwd_at(env, rb)
                if p is None or fr.empty or len(p) < 30:
                    continue
                df = pd.DataFrame({"s": _zs(p), "f": fr}).dropna()
                if len(df) > 30:
                    rho, _ = spearmanr(df["s"], df["f"])
                    ics2.append(rho)
            m = float(np.mean(ics2))
            mark = f"  -> 翻转后 IC {m:+.4f}"
        ic1[name] = m
        lines.append(f"    {name:<8} IC {m:+.4f} (n={n}){mark}")
    lines.append(f"    翻转因子: {flips if flips else '无'}")

    # ---- [1] 正交性: 财务因子 vs 现有 4 因子截面相关 ----
    lines.append("")
    lines.append("[1] 正交性: 财务因子 zscore 与现有 4 因子截面相关 (跨月均值 |corr|):")
    orth = {bf: {} for bf in BASE_F}
    corrs = {name: {bf: [] for bf in BASE_F} for name in FUNDA}
    inner = {name: {} for name in FUNDA}
    for rb in rebal:
        fdf = E.build_fdf(env, rb, None)          # 已含 9 面板 (env.panels)
        if fdf is None:
            continue
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        c = zdf.corr()
        for name in FUNDA:
            if name not in c.index:
                continue
            for bf in BASE_F:
                if bf in c.index:
                    v = c.loc[name, bf]
                    if np.isfinite(v):
                        corrs[name][bf].append(v)
            for f2 in FUNDA:
                if f2 in c.index and f2 != name:
                    v = c.loc[name, f2]
                    if np.isfinite(v):
                        inner[name].setdefault(f2, []).append(v)
    lines.append("    " + "".join(f"{bf:>10}" for bf in BASE_F) + "    内部(对财务均)")
    for name in FUNDA:
        vals = [np.mean(corrs[name][bf]) if corrs[name][bf] else np.nan for bf in BASE_F]
        iv = inner[name]
        iv_means = [np.mean(v) for v in iv.values()] if iv else []
        lines.append(f"    {name:<8}" + "".join(f"{v:>10.3f}" if np.isfinite(v) else f"{'-':>10}"
                                                for v in vals)
                     + f"    {np.mean([abs(x) for x in iv_means]):>10.3f}")
    for bf in BASE_F:
        orth[bf] = {name: (float(np.mean(corrs[name][bf])) if corrs[name][bf] else None)
                    for name in FUNDA}

    # ---- [2] 打分 IC: 基线 vs 增量组合 ----
    lines.append("")
    lines.append("[2] 打分 IC (全体成分股截面 Spearman vs 持有期收益):")
    combo_ics = {}
    ics_base = _ic_series(env, rebal, lambda e, rb: E.score_at(e, rb, None))
    st0 = _ic_stat(ics_base)
    combo_ics["base"] = st0
    lines.append(f"    base(4因子): IC {st0['mean']:+.4f} | ICIR {st0['icir']:+.3f} | 正占比 {st0['pos']:.0%} (n={st0['n']})")
    for cname, cols in COMBOS.items():
        ext = {k: panels[k] for k in cols}
        ics = _ic_series(env, rebal, lambda e, rb, _x=ext: E.score_at(e, rb, _x))
        st = _ic_stat(ics)
        combo_ics[cname] = st
        lines.append(f"    {cname:>4}(+{cols}): IC {st['mean']:+.4f} ({st['mean']-st0['mean']:+.4f}) | "
                     f"ICIR {st['icir']:+.3f} | 正占比 {st['pos']:.0%} (n={st['n']})")

    # ---- [3] 阈值选股回测 (持仓对齐) ----
    # 持仓数只取决于打分, 先纯打分缓存全期 score, 网格扫平均持仓 -> 找对齐基线档,
    # 再只回测对齐档 (避免每个阈值跑完整回测)。
    lines.append("")
    lines.append("[3] 阈值选股回测 (等权+阶段4+万1, 各组合独立重标定阈值, 持仓对齐基线 ~54.5):")
    s_base, st_base = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                     e_ovn, e_intra, use_hrp=False, use_ma20=False,
                                     st_map=st_map, limit_sets=(one_up, one_dn),
                                     tradable=tf5, score_thr=THR, ext_panels=None)
    m_base = _metrics(s_base)
    ns0 = float(np.mean(st_base["n_selected"]))
    bt = {"base": dict(thr=THR, m=m_base, n_sel=ns0)}
    lines.append(f"    base >=0.93 (持仓 {ns0:.1f}): 终值 {m_base['final']:.4f} | 年化 {m_base['ann']:.2%} | "
                 f"Sharpe {m_base['sharpe']:.2f} | MaxDD {m_base['mdd']:.2%} | 卡玛 {m_base['calmar']:.2f}")
    THR_GRID = [round(0.40 + 0.05 * i, 2) for i in range(22)]   # 0.40 ~ 1.45
    for cname, cols in COMBOS.items():
        ext = {k: panels[k] for k in cols}
        scores = {rb: E.score_at(env, rb, ext) for rb in rebal}
        prof = []
        for thr in THR_GRID:
            ns = np.mean([int((s >= thr).sum()) for s in scores.values() if s is not None]) if scores else 0.0
            prof.append((thr, float(ns)))
        thr_b = min(prof, key=lambda t: abs(t[1] - ns0))[0]
        ns_b = dict(prof)[thr_b]
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=False, use_ma20=False,
                               st_map=st_map, limit_sets=(one_up, one_dn),
                               tradable=tf5, score_thr=thr_b, ext_panels=ext)
        m_b = _metrics(s)
        bt[cname] = dict(thr=thr_b, m=m_b, n_sel=ns_b)
        lines.append(f"    {cname:>4}(+{cols}): 持仓对齐档 >= {thr_b} (持仓 {ns_b:.1f} vs 基线 {ns0:.1f}): "
                     f"终值 {m_b['final']:.4f} | 年化 {m_b['ann']:.2%} vs 基线 {m_base['ann']:.2%} | "
                     f"Sharpe {m_b['sharpe']:.2f} vs {m_base['sharpe']:.2f} | "
                     f"MaxDD {m_b['mdd']:.2%} | 卡玛 {m_b['calmar']:.2f} vs {m_base['calmar']:.2f}")
    print("\n".join(lines))

    # ---- 图1: 打分 IC 时间序列 (base vs all) ----
    ics_all = _ic_series(env, rebal, lambda e, rb: E.score_at(e, rb, {k: panels[k] for k in FUNDA}))
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    ax.plot(range(len(ics_base)), ics_base, lw=1.0, color="#333", alpha=0.7, label="base 4因子")
    ax.plot(range(len(ics_all)), ics_all, lw=1.0, color="#d62728", alpha=0.7, label="+财务全量 9因子")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axhline(st0["mean"], color="#333", ls="--", lw=1.2)
    ax.axhline(combo_ics["all"]["mean"], color="#d62728", ls="--", lw=1.2)
    ax.set_title("月度打分 IC: 基线 vs +PIT 财务因子 (全量)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp1 = os.path.join(C.OUT_DIR, "funda_factor_ic.png")
    fig.savefig(fp1, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp1}")

    # ---- 图2: Top60 散点 (base / q3 / g2 / all) ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (cname, cols) in zip(axes.ravel(), [("base", None)] + [(c, COMBOS[c]) for c in COMBOS]):
        ext = None if cname == "base" else {k: panels[k] for k in cols}
        pts = []
        for rb in rebal:
            fr = _fwd_at(env, rb)
            scored = E.score_at(env, rb, ext)
            if scored is None or len(scored) < 60:
                continue
            top = scored.sort_values(ascending=False).head(60).index
            df = pd.DataFrame({"s": scored.reindex(top), "f": fr.reindex(top)}).dropna()
            for code, row in df.iterrows():
                pts.append((row["s"], row["f"]))
        d = pd.DataFrame(pts, columns=["s", "f"])
        rho, _ = spearmanr(d["s"], d["f"]) if len(d) else (0.0, 1.0)
        ax.scatter(d["s"], d["f"] * 100, s=6, alpha=0.35, color="#444")
        bins = pd.qcut(d["s"].rank(method="first"), 10, labels=False)
        xm = d.groupby(bins)["s"].mean()
        ym = d.groupby(bins)["f"].mean() * 100
        ax.plot(xm.values, ym.values, "-o", color="orange", lw=1.5, ms=4)
        lbl = "base 4因子" if cname == "base" else f"+{cols}"
        ax.set_title(f"{cname} {lbl}: Top60 Spearman {rho:+.4f} (n={len(d)})")
        ax.set_xlabel("打分 (zscore 均值)")
        ax.set_ylabel("持有期收益 %")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fp2 = os.path.join(C.OUT_DIR, "funda_factor_scatter.png")
    fig.savefig(fp2, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp2}")

    out = dict(ic1=ic1, flips=flips, orth=orth, combo_ic=combo_ics,
               bt={k: dict(thr=v["thr"], n_sel=v["n_sel"], m=v["m"]) for k, v in bt.items()})
    with open(os.path.join(C.OUT_DIR, "funda_factor.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(C.OUT_DIR, "funda_factor.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'funda_factor.txt')} | .json")


if __name__ == "__main__":
    main()
