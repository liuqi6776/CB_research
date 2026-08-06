# -*- coding: utf-8 -*-
"""Top60 选股收益统计图 + 打分-收益散点图 (v1.1.0-wan1 口径)

图1 (results/top60_nav_stats.png): v1.1.0 净值曲线 vs 512100 ETF 基准 + 月度收益柱状
图2 (results/top60_score_return.png): 散点 x=截面打分(zscore均值), y=持有期收益
     (rb→rb_next 一个调仓月, 所有调仓日选出的 Top60 股票; 附十分位分箱均值线)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.factor_dic import run_validation as rv
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.concentration import apply_concentration, amount60_at
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df


def collect_top60(env):
    """所有调仓日 Top60 的 (rb, code, score, hold_ret) — 打分重建同 env._build_picks"""
    rows = []
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if picks is None:
            continue
        members = rv.load_index_weight(rb)
        if not members:
            continue
        fvals = {}
        for code in members:
            f1, f2, ft = env.ret_1m.get(code), env.ivol.get(code), env.turn.get(code)
            fr = env.fwd.get(code)
            if fr is None or rb not in fr.index:
                continue
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            for name in env.panels:
                p = env.panels[name].get(rb)
                if p is not None and code in p.index:
                    v = p.loc[code]
                    if np.isfinite(v):
                        row[name] = v
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < rv.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = [c for c in sf.BASE_COLS + ["VAL"] if c in zdf.columns]
        has = zdf[cols].dropna()
        if len(has) < rv.TOP_N:
            continue
        scored = has.mean(axis=1)
        for code, score in scored.nlargest(rv.TOP_N).items():
            h = env.pct_df[code].reindex(hold).dropna()
            if len(h) < 5:
                continue
            ret = (1.0 + h / 100.0).prod() - 1.0
            rows.append(dict(rb=rb, code=code, score=float(score), ret=float(ret)))
    return pd.DataFrame(rows)


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

    s11, st11 = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=True, use_ma20=False,
                               st_map=st_map, limit_sets=(one_up, one_dn),
                               tradable=tf5, concentration=_conc)
    n = len(s11)
    ann = (s11.iloc[-1] / s11.iloc[0]) ** (242.0 / (n - 1)) - 1.0
    dd = ((s11.cummax() - s11) / s11.cummax()).max()

    # ---- 图1: 净值 + 月度收益 ----
    etf = C.load_idx("512100.SH")
    base = etf["close"].reindex(s11.index).fillna(method="ffill")
    base = base / base.iloc[0]
    dts = pd.to_datetime(s11.index, format="%Y%m%d")
    m_last = s11.groupby(s11.index.str[:6]).last()
    m_ret = m_last.pct_change().dropna()
    m_years = m_ret.index.str[:4]

    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.2]})
    ax[0].plot(dts, base.values, label="512100 ETF (基准)", lw=1.0, color="#888")
    ax[0].plot(dts, s11.values, label="Top60 选股 (v1.1.0, 万1)", lw=1.4, color="#d62728")
    ax[0].set_title("Top60 选股收益统计 (BASE+VAL 四因子 + IVW120 + 阶段4/5 约束, RS12 择时)\n"
                    f"年化 {ann:.2%} | MaxDD {dd:.2%} | 终值 {s11.iloc[-1]:.4f} | "
                    f"剔除 {st11.get('n_trad_removed', 0)} 只次")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(alpha=0.3)

    colors = ["#d62728" if v < 0 else "#2ca02c" for v in m_ret.values]
    ax[1].bar(range(len(m_ret)), m_ret.values, color=colors, alpha=0.8, width=0.8)
    ax[1].axhline(0, color="#333", lw=0.8)
    ax[1].set_title(f"月度收益 (n={len(m_ret)}, 均值 {m_ret.mean():+.2%}, 胜率 {(m_ret > 0).mean():.0%})")
    step = max(1, len(m_ret) // 12)
    ax[1].set_xticks(range(0, len(m_ret), step))
    ax[1].set_xticklabels(m_ret.index[::step], rotation=45, fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fp1 = os.path.join(C.OUT_DIR, "top60_nav_stats.png")
    fig.savefig(fp1, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp1}")

    # ---- 图2: 打分-收益散点 ----
    df = collect_top60(env)
    df["year"] = df["rb"].str[:4]
    spear = df["score"].corr(df["ret"], method="spearman")
    pear = df["score"].corr(df["ret"], method="pearson")

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    sc = ax.scatter(df["score"], df["ret"], s=9, alpha=0.35,
                    c=df["year"].astype(int), cmap="viridis")
    cbar = fig.colorbar(sc, ax=ax, label="调仓年份")
    # 十分位分箱均值线
    try:
        bins = pd.qcut(df["score"], 10, duplicates="drop")
        bmean = df.groupby(bins, observed=True)["ret"].mean()
        xmid = [iv.mid for iv in bmean.index]
        ax.plot(xmid, bmean.values, color="#d62728", lw=2.0, marker="o", ms=5,
                label="十分位分箱均值")
    except Exception:
        pass
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("截面打分 (zscore 均值: ret_1m + ivol + turnover_vol_20 + VAL)")
    ax.set_ylabel("持有期收益 (调仓日 rb → 下一调仓日, 复利)")
    ax.set_title(f"Top60 选股打分 vs 持有期收益 (n={len(df)}, 调仓期 {df['rb'].nunique()} 期)\n"
                 f"Spearman {spear:+.3f} | Pearson {pear:+.3f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp2 = os.path.join(C.OUT_DIR, "top60_score_return.png")
    fig.savefig(fp2, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp2}")

    # 分箱统计输出 (控制台 + txt)
    lines = ["Top60 打分-持有期收益: 十分位分箱", "=" * 78]
    if "bmean" in dir():
        lines.append(f"{'分箱(score)':<22}{'样本':>6}{'均值收益':>12}")
        for iv, g in df.groupby(pd.qcut(df["score"], 10, duplicates="drop"), observed=True):
            lines.append(f"{str(iv):<22}{len(g):>6}{g['ret'].mean():>+12.2%}")
    lines.append(f"全体: n={len(df)} | Spearman {spear:+.3f} | Pearson {pear:+.3f} | "
                 f"收益>0 占比 {(df['ret'] > 0).mean():.0%}")
    txt = "\n".join(lines)
    print("\n" + txt)
    fp3 = os.path.join(C.OUT_DIR, "top60_score_return.txt")
    with open(fp3, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[saved] {fp3}")


if __name__ == "__main__":
    main()
