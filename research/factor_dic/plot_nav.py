# -*- coding: utf-8 -*-
"""
净值曲线图 + 逐月持仓/换手明细
重建 ENH(3因子) / ENH_F_NI(4因子+行业中性) 组合, 叠加 RS12 择时(小盘弱->持基准),
输出: results/nav_*.png(净值+回撤+RS12状态) + results/holdings_detail_*.csv(逐月持仓/换手/收益)
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
from research.factor_dic import regime_study as rs

OUT_DIR = rv.OUT_DIR
COST = rv.COST_BPS / 10000.0
SMALL_IDX = "000852.SH"


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def build_portfolio():
    """重建 ENH / ENH_F_NI 月度组合收益 + 逐月持仓, 返回 dict"""
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
    sml = load_idx(SMALL_IDX)
    etf = load_idx("512100.SH")

    port = {"ENH": [], "ENH_F_NI": []}
    bench_m, bench_etf = [], []
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
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < rv.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(rv.winsorize).apply(lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        # ENH: 3因子
        cols3 = ["ret_1m", "ivol", "turn"]
        if all(c in zdf.columns for c in cols3):
            has3 = zdf[cols3].dropna()
            if len(has3) >= rv.TOP_N:
                sc3 = has3.mean(axis=1)
                picks3 = sc3.nlargest(rv.TOP_N).index
                sub3 = pct_df.reindex(columns=picks3).reindex(hold).fillna(0.0) / 100.0
                net3 = (1 + sub3.mean(axis=1)).prod() - 1 - COST
                port["ENH"].append(dict(rb=rb, rb_next=rb_next, picks=list(picks3), net=net3))
        # ENH_F_NI: 4因子 + 行业中性
        cols4 = ["ret_1m", "ivol", "turn", "roe"]
        if all(c in zdf.columns for c in cols4):
            has4 = zdf[cols4].dropna()
            if len(has4) >= rv.TOP_N:
                ind = pd.Series({c: ind_map.get(c, "NA") for c in has4.index}, index=has4.index)
                sc4 = has4.mean(axis=1).groupby(ind).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
                picks4 = sc4.nlargest(rv.TOP_N).index
                sub4 = pct_df.reindex(columns=picks4).reindex(hold).fillna(0.0) / 100.0
                net4 = (1 + sub4.mean(axis=1)).prod() - 1 - COST
                port["ENH_F_NI"].append(dict(rb=rb, rb_next=rb_next, picks=list(picks4), net=net4))
        b = sml["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_m.append((rb, (1 + b).prod() - 1))
        be = etf["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_etf.append((rb, (1 + be).prod() - 1))
    return port, pd.Series(dict(bench_m)), pd.Series(dict(bench_etf)), sml


def rs12_signal(sml, rebal):
    """RS12: 过去12月 000852/000300 相对强度>0 (调仓日取值)"""
    big = load_idx("000300.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    return ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)


def holdings_detail(rows, name):
    """逐月持仓/换手明细 -> DataFrame"""
    out = []
    prev = None
    for r in rows:
        picks = r["picks"]
        overlap = len(set(picks) & set(prev)) if prev is not None else np.nan
        turnover = 1 - overlap / len(picks) if prev is not None else np.nan
        out.append(dict(调仓日=r["rb"], 下期调仓=r["rb_next"], 持仓数=len(picks),
                        月度净收益=r["net"], 与上期重叠=overlap, 换手率=turnover,
                        持仓=",".join(picks)))
        prev = picks
    return pd.DataFrame(out)


def plot_nav(rows, bm_idx, bm_etf, sig, name, fname):
    pr = pd.Series({r["rb"]: r["net"] for r in rows})
    idx = pr.index
    nav = pr.add(1).cumprod()
    nav_t = pr.where(sig.reindex(idx), bm_etf.reindex(idx)).add(1).cumprod()
    nav_i = pr.where(sig.reindex(idx), bm_idx.reindex(idx)).add(1).cumprod()
    nav_bm = bm_idx.reindex(idx).add(1).cumprod()
    nav_etf = bm_etf.reindex(idx).add(1).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    x = np.arange(len(idx))
    ax.plot(x, nav.values, label=f"{name} 无择时", lw=1.2, color="#888")
    ax.plot(x, nav_t.values, label=f"{name}+RS12择时(vs ETF512100)", lw=2.0, color="#c00")
    ax.plot(x, nav_i.values, label=f"{name}+RS12择时(vs 000852)", lw=1.5, color="#e07b00")
    ax.plot(x, nav_bm.values, label="基准 000852", lw=1.2, ls="--", color="#333")
    ax.plot(x, nav_etf.values, label="基准 512100ETF", lw=1.2, ls="--", color="#06c")
    ax.set_yscale("log")
    ax.set_ylabel("净值(对数)")
    ax.set_title(f"{name} + RS12 择时净值曲线 (2020~2026, 月度调仓, 20bps)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(x, 0, 1, where=sig.reindex(idx).values, color="#0c6", alpha=0.6, label="RS12>0 (小盘强→持因子)")
    ax2.fill_between(x, 0, 1, where=~sig.reindex(idx).values, color="#ccc", alpha=0.6, label="RS12<=0 (小盘弱→持基准)")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("月度调仓序号")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fp = os.path.join(OUT_DIR, fname)
    plt.savefig(fp, dpi=130)
    plt.close()
    print(f"[保存图] {fp}")
    return fp


def main():
    port, bm_idx, bm_etf, sml = build_portfolio()
    rebal = [r["rb"] for r in port["ENH"]]
    sig = rs12_signal(sml, rebal)

    for name in ["ENH", "ENH_F_NI"]:
        rows = port[name]
        if not rows:
            continue
        det = holdings_detail(rows, name)
        det.to_csv(os.path.join(OUT_DIR, f"holdings_detail_{name}.csv"), index=False, encoding="utf-8-sig")
        print(f"[保存明细] holdings_detail_{name}.csv: {len(det)}期, 平均换手率 "
              f"{det['换手率'].dropna().mean():.1%}")
        plot_nav(rows, bm_idx, bm_etf, sig, name, f"nav_{name}_rs12.png")

    # 汇总: 无择时 vs 择时
    print("\n[汇总] (年化/Sharpe/MaxDD/月胜率/超额vs ETF512100)")
    for name in ["ENH", "ENH_F_NI"]:
        pr = pd.Series({r["rb"]: r["net"] for r in port[name]})
        bm = bm_etf.reindex(pr.index)
        for tag, p in [("无择时", pr), ("RS12择时", pr.where(sig.reindex(pr.index), bm))]:
            navs = p.add(1).cumprod()
            years = len(p) / 12
            cagr = navs.iloc[-1] ** (1 / years) - 1
            sharpe = p.mean() / p.std(ddof=1) * np.sqrt(12)
            mdd = (navs.cummax() - navs).max()
            ex = (1 + p).prod() / (1 + bm).prod() - 1
            print(f"  {name:<10} {tag:<8} 年化{cagr:>7.2%} Sharpe {sharpe:.2f} MaxDD {mdd:>7.2%} 月胜率{(p>0).mean():.1%} 超额{ex:>7.2%}")


if __name__ == "__main__":
    main()
