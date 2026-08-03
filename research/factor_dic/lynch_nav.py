# -*- coding: utf-8 -*-
"""
PEG(LYNCH) vs ENH vs 混合策略(POOL_NI) 净值对比图
- 上图: 全期 2020-2026 (77月): LYNCH+RS12 vs ENH+RS12 vs 基准
- 下图: 2023-2026 同窗口 (38月): LYNCH+RS12 vs ENH_F_NI+RS12 vs POOL_NI(5因子含PEG)+RS12 vs 基准
输出: results/lynch_nav.png
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
from research.factor_dic import lynch_factor as lf

OUT_DIR = rv.OUT_DIR
COST = rv.COST_BPS / 10000.0
SMALL_IDX = "000852.SH"


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


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
    roe_pit = cb.build_roe_pit(rebal)
    ind_map = cb.load_industry_map()
    pe_map = lf.load_pe_ttm(rebal, all_codes)
    yoy_map = lf.build_yoy_pit(rebal, all_codes)
    peg = lf.build_peg(pe_map, yoy_map, rebal, all_codes)
    print(f"[load] PEG 面板 {len(peg)} 个月")

    sml = load_idx(SMALL_IDX)
    big = load_idx("000300.SH")
    etf = load_idx("512100.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    port = {k: [] for k in ["LYNCH", "ENH", "ENH_F_NI", "POOL_NI"]}
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
            if rb in roe_pit and code in roe_pit[rb].index:
                row["roe"] = roe_pit[rb].loc[code]
            if peg.get(rb) and code in peg[rb]:
                pv = -peg[rb][code]
                if np.isfinite(pv):
                    row["peg"] = pv
            if len(row) >= 1:
                fvals[code] = row
        if len(fvals) < rv.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(rv.winsorize).apply(lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        ind = pd.Series({c: ind_map.get(c, "NA") for c in fdf.index}, index=fdf.index)

        def _run(name, cols, neut):
            if not all(c in zdf.columns for c in cols):
                return
            has = zdf[cols].dropna()
            if len(has) < rv.TOP_N:
                return
            sc = has.mean(axis=1)
            if neut:
                sc = sc.groupby(ind).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
            picks = sc.nlargest(rv.TOP_N).index
            sub = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
            port[name].append(dict(rb=rb, net=(1 + sub.mean(axis=1)).prod() - 1 - COST))

        _run("LYNCH", ["peg"], False)
        _run("ENH", ["ret_1m", "ivol", "turn"], False)
        _run("ENH_F_NI", ["ret_1m", "ivol", "turn", "roe"], True)
        _run("POOL_NI", ["ret_1m", "ivol", "turn", "roe", "peg"], True)
        b = sml["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_m[rb] = (1 + b).prod() - 1
        be = etf["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_etf[rb] = (1 + be).prod() - 1

    bm_i = pd.Series(bench_m)
    bm_e = pd.Series(bench_etf)

    def nav_pr(rows):
        return pd.Series({r["rb"]: r["net"] for r in rows}).sort_index()

    def nav_timed(rows, sig):
        pr = nav_pr(rows)
        return pr.where(sig.reindex(pr.index), bm_e.reindex(pr.index))

    pr_enh, pr_lyn, pr_ni, pr_pool = (nav_timed(port[n], sig_rs12)
                                      for n in ["ENH", "LYNCH", "ENH_F_NI", "POOL_NI"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=False,
                             gridspec_kw={"height_ratios": [3, 3], "hspace": 0.18})

    def plot_axis(ax, series_dict, title, sig, with_bench=True):
        x_all = sorted(set().union(*[list(s.index) for s in series_dict.values()]))
        for lbl, s in series_dict.items():
            ax.plot(np.arange(len(x_all)), (1 + s.reindex(x_all).fillna(0)).cumprod().values,
                    label=lbl, lw=1.8)
        ax.set_yscale("log")
        ax.set_ylabel("净值(对数)")
        ax.set_title(title, fontsize=12)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.fill_between(np.arange(len(x_all)), 0, 1, where=sig.reindex(x_all).fillna(False).values,
                         color="#0c6", alpha=0.15)
        ax2.set_yticks([])
        ax2.set_ylim(0, 1)

    # 上图: 全期
    x_all = sorted(set(pr_enh.index) | set(pr_lyn.index))
    plot_axis(axes[0],
              {"ENH 3因子 +RS12择时": pr_enh,
               "LYNCH PEG +RS12择时": pr_lyn,
               "基准 000852": bm_i.reindex(x_all),
               "基准 512100ETF": bm_e.reindex(x_all)},
              "全期 2020-2026: PEG(LYNCH) vs 现有因子(ENH), 均叠加 RS12 择时(小盘弱→持ETF)",
              sig_rs12)
    # 下图: 2023-2026 同窗口
    x38 = sorted(set(pr_ni.index) | set(pr_pool.index) | set(pr_lyn.index))
    plot_axis(axes[1],
              {"LYNCH PEG +RS12": pr_lyn.reindex(x38),
               "ENH_F_NI 4因子 +RS12": pr_ni.reindex(x38),
               "POOL_NI 5因子含PEG +RS12": pr_pool.reindex(x38),
               "基准 000852": bm_i.reindex(x38),
               "基准 512100ETF": bm_e.reindex(x38)},
              "2023-2026 同窗口: 混合策略(POOL_NI, 含PEG) vs 4因子(ENH_F_NI) vs PEG(LYNCH), 均叠加 RS12 择时",
              sig_rs12)

    fp = os.path.join(OUT_DIR, "lynch_nav.png")
    plt.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[保存图] {fp}")

    # 汇总表
    print(f"\n{'策略':<28}{'区间':>7}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'月胜率':>8}{'超额vETF':>10}")
    for nm, pr in [("LYNCH+RS12", pr_lyn), ("ENH+RS12", pr_enh),
                   ("ENH_F_NI+RS12", pr_ni), ("POOL_NI+RS12", pr_pool)]:
        p = pr.dropna()
        if p.empty:
            continue
        navs = (1 + p).cumprod()
        years = len(p) / 12.0
        cagr = navs.iloc[-1] ** (1 / years) - 1
        ex = (1 + p).prod() / (1 + bm_e.reindex(p.index)).prod() - 1
        print(f"{nm:<28}{str(p.index[0])[:6]+'-'+str(p.index[-1])[:6]:>7}"
              f"{cagr:>8.2%}{p.mean()/p.std(ddof=1)*np.sqrt(12):>8.2f}"
              f"{(navs.cummax()-navs).max():>9.2%}{(p>0).mean():>8.1%}{ex:>10.2%}")


if __name__ == "__main__":
    main()
