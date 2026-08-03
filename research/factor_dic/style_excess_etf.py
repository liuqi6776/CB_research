# -*- coding: utf-8 -*-
"""
最优策略(BASE+VAL+RS12) vs 现状(BASE+RS12) vs BASE+SP+RS12 —— 相对 512100 ETF 的超额累计曲线
超额 = 策略月收益 - ETF 月收益, 累计 (1+超额).cumprod()-1
输出: results/style_excess_etf.png
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

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    val_map = sf.load_valuation(rebal, all_codes)
    funda_map = sf.build_funda_pit(rebal, all_codes)
    panels = sf.build_factors(val_map, funda_map, rebal)

    sml = sf.load_idx("000852.SH")
    big = sf.load_idx("000300.SH")
    etf = sf.load_idx("512100.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    names = ["BASE", "BASE+VAL", "BASE+SP"]
    port = {k: [] for k in names}
    bm_e = {}
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

        def _run(name, cols):
            if not all(c in zdf.columns for c in cols):
                return
            has = zdf[cols].dropna()
            if len(has) < rv.TOP_N:
                return
            picks = has.mean(axis=1).nlargest(rv.TOP_N).index
            sub = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
            port[name].append(dict(rb=rb, net=(1 + sub.mean(axis=1)).prod() - 1 - COST))

        _run("BASE", sf.BASE_COLS)
        for nm in ["VAL", "SP"]:
            _run(f"BASE+{nm}", sf.BASE_COLS + [nm])
        be = etf["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bm_e[rb] = (1 + be).prod() - 1

    bm_e = pd.Series(bm_e)

    def nav_timed(rows):
        pr = pd.Series({r["rb"]: r["net"] for r in rows}).sort_index()
        return pr.where(sig_rs12.reindex(pr.index), bm_e.reindex(pr.index))

    prs = {n: nav_timed(port[n]) for n in names}
    x_all = sorted(bm_e.index)

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {"BASE": "#888", "BASE+VAL": "#c33", "BASE+SP": "#e90"}
    for n in names:
        ex = (1 + prs[n].reindex(x_all).fillna(0)) / (1 + bm_e.reindex(x_all).fillna(0)) - 1
        cum = (1 + ex).cumprod() - 1
        ax.plot(np.arange(len(x_all)), cum.values * 100, label=f"{n} +RS12 超额 vETF",
                lw=1.8, color=colors[n])
        final = cum.iloc[-1] * 100
        ax.annotate(f"{final:+.0f}%", (len(x_all) - 1, final),
                    textcoords="offset points", xytext=(6, 0), fontsize=9, color=colors[n])
    ax.axhline(0, color="#000", lw=0.8)
    ax.set_ylabel("累计超额收益 vs 512100ETF (%)")
    ax.set_title("最优策略超额曲线: BASE+VAL+RS12 vs 现状 vs BASE+SP, 相对 512100ETF (2020-2026)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(0, len(x_all), 12),
                  [d[:6] for d in x_all[::12]])

    fp = os.path.join(OUT_DIR, "style_excess_etf.png")
    plt.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[保存图] {fp}")

    print(f"\n{'策略':<20}{'累计超额vETF':>14}{'年化超额':>10}")
    for n in names:
        ex = (1 + prs[n].reindex(x_all).fillna(0)) / (1 + bm_e.reindex(x_all).fillna(0)) - 1
        cum = (1 + ex).cumprod() - 1
        cagr_ex = (1 + cum.iloc[-1]) ** (1 / (len(prs[n]) / 12.0)) - 1
        print(f"{n:<20}{cum.iloc[-1]:>14.1%}{cagr_ex:>10.2%}")


if __name__ == "__main__":
    main()
