# -*- coding: utf-8 -*-
"""
画 512100 ETF vs 最优策略 (BASE+VAL+RS12+MA20三档0.98) 净值收益曲线对比 (2020-2026)
输出: results/nav_vs_etf.png
"""
import os
import sys
import time

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
TOP_N = rv.TOP_N
DEEP = 0.98


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def stats(pr):
    pr = pr.dropna()
    navs_c = (1 + pr).cumprod()
    years = len(pr) / 12.0
    cagr = navs_c.iloc[-1] ** (1 / years) - 1
    sharpe = pr.mean() / pr.std(ddof=1) * np.sqrt(12)
    mdd = ((navs_c.cummax() - navs_c) / navs_c.cummax()).max()
    return cagr, sharpe, mdd


def main():
    t0 = time.time()
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

    sml = load_idx("000852.SH")
    big = load_idx("000300.SH")
    etf = load_idx("512100.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    idx_close = sml["close"]
    ma20 = idx_close.rolling(20).mean()
    # 修复同日信号前视: T-1 日收盘信号, T 日生效
    idx_close_1 = idx_close.shift(1)
    ma20_1 = ma20.shift(1)
    etf_ret = etf["pct_chg"] / 100.0

    # ---------- 每月 Top50 (BASE+VAL) ----------
    picks_map = {}
    for rb in rebal:
        members = rv.load_index_weight(rb)
        if members is None:
            continue
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
        if len(fvals) < TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = sf.BASE_COLS + ["VAL"]
        has = zdf[cols].dropna()
        if len(has) < TOP_N:
            continue
        picks_map[rb] = has.mean(axis=1).nlargest(TOP_N).index.tolist()
    print(f"[load] Top50 月份 {len(picks_map)}, 耗时 {time.time()-t0:.0f}s", flush=True)

    # ---------- 日频回测 (MA20 三档 0.98) ----------
    nav_rb = {rebal[0]: 1.0}
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        rb_next = rebal[i + 1]
        if rb not in picks_map:
            nav_rb[rb_next] = nav_rb.get(rb, 1.0)
            continue
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        picks = picks_map[rb]
        comb = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
        comb_ret = comb.mean(axis=1)
        e_ret = etf_ret.reindex(hold).fillna(0.0)
        rs12_on = bool(sig_rs12.loc[rb]) if rb in sig_rs12.index else True
        nav = nav_rb.get(rb, 1.0)
        for t in hold:
            r_t = e_ret.loc[t]
            if rs12_on:
                w = 1.0
                c, m = idx_close_1.get(t, np.nan), ma20_1.get(t, np.nan)
                if np.isfinite(c) and np.isfinite(m):
                    w = 1.0 if c >= m else (0.5 if c >= DEEP * m else 0.0)
                r_t = w * comb_ret.loc[t]
            nav *= (1.0 + r_t)
        nav *= (1.0 - COST)
        nav_rb[rb_next] = nav
    strat_nav = pd.Series(nav_rb).sort_index()

    # ETF 月净值 (与 risk_control_bt.py 同口径: 完整月度序列, 键=调仓日 rb, 含首月, 补初始1.0)
    etf_m = {}
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        rb_next = rebal[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        etf_m[rb] = (1 + etf_ret.reindex(hold).fillna(0.0)).prod() - 1
    etf_pr = pd.Series(etf_m).sort_index()
    etf_nav = pd.Series(np.concatenate([[1.0], np.cumprod(1 + etf_pr.values)]),
                        index=rebal)

    pr_s = strat_nav.pct_change().dropna()
    pr_e = etf_nav.pct_change().dropna()
    c_s = stats(pr_s)
    c_e = stats(pr_e)

    x_all = strat_nav.index
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.plot(np.arange(len(x_all)), strat_nav.values, label="最优策略 (BASE+VAL+RS12+MA20三档)", lw=2.0, color="#c33")
    ax.plot(np.arange(len(x_all)), etf_nav.reindex(x_all).ffill().values, label="512100 ETF (中证1000)", lw=1.6, ls="--", color="#36c")
    ax.set_yscale("log")
    ax.set_ylabel("净值(对数)")
    ax.set_title("512100 ETF vs 最优策略 净值对比 (2020-2026, 月度调仓 Top50)", fontsize=13)

    # 底部统计标签
    txt = (f"策略: 年化 {c_s[0]:.1%} | Sharpe {c_s[1]:.2f} | MaxDD {c_s[2]:.1%}\n"
           f"ETF : 年化 {c_e[0]:.1%} | Sharpe {c_e[1]:.2f} | MaxDD {c_e[2]:.1%}")
    ax.text(0.985, 0.05, txt, transform=ax.transAxes, fontsize=10,
            va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.9))

    # RS12 强弱带
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(len(x_all)), 0, 1,
                     where=sig_rs12.reindex(x_all).fillna(False).values, color="#0c6", alpha=0.15)
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)

    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    fp = os.path.join(OUT_DIR, "nav_vs_etf.png")
    plt.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[保存图] {fp}   (总耗时 {time.time()-t0:.0f}s)")
    print(f"策略: 年化 {c_s[0]:.2%} Sharpe {c_s[1]:.2f} MaxDD {c_s[2]:.2%}")
    print(f"ETF : 年化 {c_e[0]:.2%} Sharpe {c_e[1]:.2f} MaxDD {c_e[2]:.2%}")


if __name__ == "__main__":
    main()
