# -*- coding: utf-8 -*-
"""三个变体 vs 中证1000 ETF (512100) 收益曲线对比

- A: v1.1.0 基线 (Top60 + IVW120 + 阶段4/5, 万1)
- B: >=0.93 等权 (阶段4, 无集中度, 万1)         <- 当前探索最优
- C: >=0.93 等权 + 阶段5 集中度 (万1)
- ETF: 512100 中证1000 ETF (同起点归一化)

输出: results/abc_vs_etf_nav.png + results/abc_vs_etf.txt
"""
import json
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

from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.concentration import apply_concentration, amount60_at
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df

THR = 0.93


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

    def _run(use_hrp, with_conc):
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=use_hrp, use_ma20=False,
                               st_map=st_map, limit_sets=(one_up, one_dn),
                               tradable=tf5, concentration=_conc if with_conc else None,
                               score_thr=None if use_hrp else THR)
        return s, st

    runs = {
        "A v1.1.0 基线 (IVW120+集中度)": _run(use_hrp=True, with_conc=True),
        "B >=0.93 等权": _run(use_hrp=False, with_conc=False),
        "C >=0.93 等权+集中度": _run(use_hrp=False, with_conc=True),
    }

    dts = pd.to_datetime(runs["A v1.1.0 基线 (IVW120+集中度)"][0].index, format="%Y%m%d")
    str_idx = [d.strftime("%Y%m%d") for d in dts]

    # 512100 中证1000 ETF 净值 (pct_chg 累乘, 复权口径; 与策略 nav 同日期序列对齐)
    etf_ret = C.load_idx("512100.SH")["pct_chg"].astype(float) / 100.0
    etf_series = (1.0 + etf_ret.reindex(str_idx).fillna(0.0)).cumprod()

    lines = ["三个变体 vs 中证1000 ETF (512100) — 收益曲线对比 (2020-01~2026-07, 万1)", "=" * 88]
    hdr = f"{'变体':<28}{'终值':>9}{'年化':>9}{'Sharpe':>8}{'MaxDD':>9}{'卡玛':>7}{'超额(年化)':>10}"
    lines.append(hdr)
    metrics = {}
    m_etf = _metrics(etf_series)
    for name, (s, st) in runs.items():
        m = _metrics(s)
        metrics[name] = m
        lines.append(f"{name:<28}{m['final']:>9.4f}{m['ann']:>9.2%}{m['sharpe']:>8.2f}"
                     f"{m['mdd']:>9.2%}{m['calmar']:>7.2f}{m['ann']-m_etf['ann']:>+10.2%}")
    lines.append(f"{'512100 中证1000 ETF':<28}{m_etf['final']:>9.4f}{m_etf['ann']:>9.2%}"
                 f"{m_etf['sharpe']:>8.2f}{m_etf['mdd']:>9.2%}{m_etf['calmar']:>7.2f}{'—':>10}")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(12.5, 7))
    styles = [
        ("A v1.1.0 基线 (IVW120+集中度)", "#333", 1.7, "-"),
        ("B >=0.93 等权", "#d62728", 1.9, "-"),
        ("C >=0.93 等权+集中度", "#1f77b4", 1.5, "-"),
    ]
    for name, c, lw, ls in styles:
        s = runs[name][0]
        m = metrics[name]
        ax.plot(dts, s.values, lw=lw, ls=ls, color=c,
                label=f"{name} | 终值 {m['final']:.3f} | 年化 {m['ann']:.1%} | 卡玛 {m['calmar']:.2f}")
    m = m_etf
    ax.plot(dts, etf_series.values, lw=1.6, ls="--", color="#2ca02c",
            label=f"512100 中证1000 ETF | 终值 {m['final']:.3f} | 年化 {m['ann']:.1%}")
    ax.axhline(1.0, color="#888", lw=0.8)
    ax.set_title("三变体 vs 中证1000 ETF 收益曲线 (2020-01~2026-07, 万1, RS12 择时)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "abc_vs_etf_nav.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    out = {name: dict(metrics=metrics[name]) for name in metrics}
    out["ETF"] = dict(metrics=m_etf)
    with open(os.path.join(C.OUT_DIR, "abc_vs_etf.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(C.OUT_DIR, "abc_vs_etf.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'abc_vs_etf.txt')} | .json")


if __name__ == "__main__":
    main()
