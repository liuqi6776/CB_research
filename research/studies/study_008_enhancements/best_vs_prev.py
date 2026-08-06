# -*- coding: utf-8 -*-
"""当前最优 vs 之前实盘最优 — 回测曲线对比 (净值 + 回撤双面板)

- 之前实盘最优 (v1.1.0 基线): Top60 + IVW120 + 阶段4 过滤 + 阶段5 集中度 + RS12 + 万1
- 当前探索最优 (>=0.93 等权): score>=0.93 全选等权 + 阶段4 过滤 (无集中度) + RS12 + 万1

输出: results/best_vs_prev_nav.png (上: 净值, 下: 回撤) + results/best_vs_prev.txt
"""
import os
import sys

import numpy as np

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

    # 之前实盘最优: v1.1.0 基线
    s_prev, st_prev = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                     e_ovn, e_intra, use_hrp=True, use_ma20=False,
                                     st_map=st_map, limit_sets=(one_up, one_dn),
                                     tradable=tf5, concentration=_conc)
    # 当前探索最优: >=0.93 等权
    s_best, st_best = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                     e_ovn, e_intra, use_hrp=False, use_ma20=False,
                                     st_map=st_map, limit_sets=(one_up, one_dn),
                                     tradable=tf5, score_thr=THR)

    m_prev, m_best = _metrics(s_prev), _metrics(s_best)
    dts = __import__("pandas").to_datetime(s_prev.index, format="%Y%m%d")

    lines = ["当前最优 vs 之前实盘最优 — 回测曲线对比 (万1, RS12 择时)", "=" * 84]
    hdr = f"{'方式':<30}{'终值':>9}{'年化':>9}{'Sharpe':>8}{'MaxDD':>9}{'卡玛':>7}{'月胜率':>8}"
    lines.append(hdr)
    lines.append(f"{'v1.1.0 基线 (Top60+IVW120+集中度)':<30}{m_prev['final']:>9.4f}"
                 f"{m_prev['ann']:>9.2%}{m_prev['sharpe']:>8.2f}{m_prev['mdd']:>9.2%}"
                 f"{m_prev['calmar']:>7.2f}{m_prev['m_win']:>8.0%}")
    lines.append(f"{f'>=0.93 等权 (分数阈值全选)':<30}{m_best['final']:>9.4f}"
                 f"{m_best['ann']:>9.2%}{m_best['sharpe']:>8.2f}{m_best['mdd']:>9.2%}"
                 f"{m_best['calmar']:>7.2f}{m_best['m_win']:>8.0%}")
    lines.append(f"\n差额: 年化 {m_best['ann']-m_prev['ann']:+.2%} | "
                 f"Sharpe {m_best['sharpe']-m_prev['sharpe']:+.2f} | "
                 f"MaxDD {m_best['mdd']-m_prev['mdd']:+.2%} | 终值 {m_best['final']-m_prev['final']:+.4f}")
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    # 上: 净值
    ax = axes[0]
    ax.plot(dts, s_prev.values, lw=1.8, color="#333",
            label=f"v1.1.0 基线 (之前实盘最优): 终值 {m_prev['final']:.4f} | "
                  f"年化 {m_prev['ann']:.2%} | 卡玛 {m_prev['calmar']:.2f}")
    ax.plot(dts, s_best.values, lw=1.8, color="#d62728",
            label=f">=0.93 等权 (当前探索最优): 终值 {m_best['final']:.4f} | "
                  f"年化 {m_best['ann']:.2%} | 卡玛 {m_best['calmar']:.2f}")
    ax.set_title("≥0.93 分数阈值等权 vs v1.1.0 实盘基线 — 回测净值 (2020-01~2026-07, 万1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    # 下: 回撤
    ax2 = axes[1]
    dd_prev = s_prev / s_prev.cummax() - 1.0
    dd_best = s_best / s_best.cummax() - 1.0
    ax2.plot(dts, dd_prev.values * 100, lw=1.3, color="#333",
             label=f"基线 MaxDD {m_prev['mdd']:.1%}")
    ax2.plot(dts, dd_best.values * 100, lw=1.3, color="#d62728",
             label=f">=0.93 等权 MaxDD {m_best['mdd']:.1%}")
    ax2.fill_between(dts, dd_best.values * 100, 0, color="#d62728", alpha=0.15)
    ax2.set_ylabel("回撤 %")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "best_vs_prev_nav.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    with open(os.path.join(C.OUT_DIR, "best_vs_prev.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'best_vs_prev.txt')}")


if __name__ == "__main__":
    main()
