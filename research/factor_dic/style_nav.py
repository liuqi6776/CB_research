# -*- coding: utf-8 -*-
"""
风格因子净值对比图: BASE(3因子) vs BASE+VAL / BASE+SP / BASE+BP / BASE+DP
全部叠加 RS12 择时(小盘弱→512100ETF), vs 基准 000852 / 512100ETF
输出: results/style_nav.png
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic import style_factors as sf

OUT_DIR = rv.OUT_DIR
COST = rv.COST_BPS / 10000.0


def main():
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]
    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"[load] 调仓日 {len(rebal)}, 成分股 {len(all_codes)}", flush=True)

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    ind_map = cb.load_industry_map()
    val_map = sf.load_valuation(rebal, all_codes)
    funda_map = sf.build_funda_pit(rebal, all_codes)
    panels = sf.build_factors(val_map, funda_map, rebal)

    sml = sf.load_idx("000852.SH")
    big = sf.load_idx("000300.SH")
    etf = sf.load_idx("512100.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    names = ["BASE", "BASE+VAL", "BASE+SP", "BASE+BP", "BASE+DP"]
    port = {k: [] for k in names}
    bench_m, bench_etf = {}, {}
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        rb_next = rebal[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        fvals = {}
        for code in members:
            f1, f2, ft = ret_1m.get(code), ivol.get(code), turn.get(code)
            fr = fwd.get(code)
            if fr is None or rb not in fr.index:
                continue
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            for name in panels:
                p = panels[name].get(rb)
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
        ind = pd.Series({c: ind_map.get(c, "NA") for c in fdf.index}, index=fdf.index)

        def _run(name, cols):
            if not all(c in zdf.columns for c in cols):
                return
            has = zdf[cols].dropna()
            if len(has) < rv.TOP_N:
                return
            sc = has.mean(axis=1)
            picks = sc.nlargest(rv.TOP_N).index
            sub = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
            port[name].append(dict(rb=rb, net=(1 + sub.mean(axis=1)).prod() - 1 - COST))

        _run("BASE", sf.BASE_COLS)
        for nm in ["VAL", "SP", "BP", "DP"]:
            _run(f"BASE+{nm}", sf.BASE_COLS + [nm])
        b = sml["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_m[rb] = (1 + b).prod() - 1
        be = etf["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_etf[rb] = (1 + be).prod() - 1

    bm_i = pd.Series(bench_m)
    bm_e = pd.Series(bench_etf)

    def nav_pr(rows):
        return pd.Series({r["rb"]: r["net"] for r in rows}).sort_index()

    def nav_timed(rows):
        pr = nav_pr(rows)
        return pr.where(sig_rs12.reindex(pr.index), bm_e.reindex(pr.index))

    prs = {n: nav_timed(port[n]) for n in names}
    x_all = sorted(bm_e.index)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    colors = {"BASE": "#888", "BASE+VAL": "#c33", "BASE+SP": "#e90",
              "BASE+BP": "#36c", "BASE+DP": "#282"}
    for n in names:
        ax.plot(np.arange(len(x_all)), (1 + prs[n].reindex(x_all).fillna(0)).cumprod().values,
                label=f"{n} +RS12择时", lw=1.8, color=colors[n])
    ax.plot(np.arange(len(x_all)), (1 + bm_i.reindex(x_all).fillna(0)).cumprod().values,
            label="基准 000852", lw=1.2, ls="--", color="#666")
    ax.plot(np.arange(len(x_all)), (1 + bm_e.reindex(x_all).fillna(0)).cumprod().values,
            label="基准 512100ETF", lw=1.2, ls="--", color="#999")
    ax.set_yscale("log")
    ax.set_ylabel("净值(对数)")
    ax.set_title("风格因子增量: BASE(3因子) 叠加价值系(SP/BP/DP/VAL), 均+RS12择时 (2020-2026)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(len(x_all)), 0, 1,
                     where=sig_rs12.reindex(x_all).fillna(False).values,
                     color="#0c6", alpha=0.15)
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)

    fp = os.path.join(OUT_DIR, "style_nav.png")
    plt.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[保存图] {fp}")

    print(f"\n{'策略':<20}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'超额vETF':>10}")
    for n in names:
        p = prs[n].dropna()
        navs = (1 + p).cumprod()
        years = len(p) / 12.0
        ex = (1 + p).prod() / (1 + bm_e.reindex(p.index)).prod() - 1
        print(f"{n:<20}{navs.iloc[-1]**(1/years)-1:>8.2%}{p.mean()/p.std(ddof=1)*np.sqrt(12):>8.2f}"
              f"{(navs.cummax()-navs).max():>9.2%}{ex:>10.2%}")


if __name__ == "__main__":
    main()
