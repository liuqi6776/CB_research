# -*- coding: utf-8 -*-
"""佣金费率万1 重测 (v1.1.0 全链: 阶段4 可交易过滤含ST剔除 + 阶段5 集中度约束, ETF腿)

对照: 佣金万1 (用户实盘费率) vs 万2.5 (原保守值)
输出: results/v11_fee_compare.txt/.json/.png (净值曲线 + 回撤)
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
from research.studies.study_008_enhancements import risk_control_real as RC
from research.studies.study_008_enhancements.concentration import apply_concentration, amount60_at
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df


def _metrics(s):
    n = len(s)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
    ret = s.pct_change().dropna()
    shp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(242.0)
    dd = ((s.cummax() - s) / s.cummax()).max()
    return dict(final=float(s.iloc[-1]), ann=cagr, sharpe=shp, mdd=float(dd),
                calmar=float(cagr / dd) if dd > 0 else 0.0)


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

    def run():
        return E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                              e_ovn, e_intra, use_hrp=True, use_ma20=False,
                              st_map=st_map, limit_sets=(one_up, one_dn),
                              tradable=tf5, concentration=_conc)

    s1, st1 = run()                     # 万1 (当前默认)
    RC.COMMISSION = 0.00025             # 万2.5 对照
    E.ETF_FEE = 0.00025
    s25, st25 = run()
    RC.COMMISSION = 0.0001
    E.ETF_FEE = 0.0001

    m1, m25 = _metrics(s1), _metrics(s25)
    lines = ["佣金费率重测: v1.1.0 全链 (阶段4 ST剔除 + 阶段5 集中度, ETF腿)", "=" * 100]
    lines.append(f"{'指标':<10}{'万1(实盘费率)':>16}{'万2.5(原保守)':>16}{'差异':>12}")
    for k, lab in [("final", "终值"), ("ann", "年化"), ("sharpe", "Sharpe"),
                   ("mdd", "MaxDD"), ("calmar", "卡玛")]:
        lines.append(f"{lab:<10}{m1[k]:>16.4f}{m25[k]:>16.4f}{m1[k]-m25[k]:>+12.4f}")
    lines.append(f"  调仓成本 万1: {st1['turn']:.2%} NAV (买 {st1['cost_buy']:.2%} + 卖 {st1['cost_sell']:.2%}) "
                 f"| 万2.5: {st25['turn']:.2%} NAV")
    lines.append(f"  ST/流动性剔除 {st1['n_trad_removed']} 只次 | n_missing {st1['n_missing']} 月 | "
                 f"阻塞 买{st1['n_buy_block']} 停牌{st1['n_suspend']} (两费率一致)")
    print("\n".join(lines))

    # 曲线图
    dts = pd.to_datetime(s1.index, format="%Y%m%d")
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(dts, s25.values, label="佣金万2.5 (原保守)", lw=1.1, color="#888")
    ax[0].plot(dts, s1.values, label="佣金万1 (实盘费率)", lw=1.4, color="#d62728")
    ax[0].set_title("v1.1.0 净值曲线 (阶段4 ST剔除 + 阶段5 集中度, 防御腿=ETF 512100)\n"
                    f"万1: 年化{m1['ann']:.2%} Sharpe {m1['sharpe']:.2f} MaxDD {m1['mdd']:.2%} 终值 {m1['final']:.4f} | "
                    f"万2.5: 年化{m25['ann']:.2%} 终值 {m25['final']:.4f}")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(alpha=0.3)
    dd1 = 1.0 - s1 / s1.cummax()
    dd25 = 1.0 - s25 / s25.cummax()
    ax[1].plot(dts, dd25.values, color="#888", lw=1.0)
    ax[1].plot(dts, dd1.values, color="#d62728", lw=1.2)
    ax[1].fill_between(dts, 0, dd1.values, alpha=0.25, color="#d62728")
    ax[1].set_title("回撤 (万1)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fp_png = os.path.join(C.OUT_DIR, "v11_fee_compare.png")
    fig.savefig(fp_png, dpi=130)
    plt.close(fig)

    fp = os.path.join(C.OUT_DIR, "v11_fee_compare.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "v11_fee_compare.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fee_1e4=dict(metrics=m1, **{k: int(v) for k, v in st1.items()}),
                       fee_2p5e4=dict(metrics=m25, **{k: int(v) for k, v in st25.items()})),
                  f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {fp} | {fp_png}")


if __name__ == "__main__":
    main()
