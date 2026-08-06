# -*- coding: utf-8 -*-
"""阶段4+5 合入基线对照: 冻结 v1.0.0 vs v1.1.0

v1.0.0: engine.run_backtest 默认口径 (IVW120 + 阶段2 费率 + fail-closed)
v1.1.0: + 阶段4 可交易过滤 (订单名单: ST/退市停牌/低流动性/波动率下限12%)
        + 阶段5 集中度约束 (单股4% / 行业20% / Top5 20% / 容量5%×ADTV60)
输出: results/v11_compare.txt | .json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.concentration import apply_concentration, amount60_at
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df


def _metrics(s):
    n = len(s)
    ret = s.pct_change().dropna()
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
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

    kwargs = dict(use_hrp=True, use_ma20=False, st_map=st_map,
                  limit_sets=(one_up, one_dn))
    s10, st10 = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, **kwargs)
    s11, st11 = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, **kwargs,
                               tradable=tf5, concentration=_conc)

    m10, m11 = _metrics(s10), _metrics(s11)
    both = pd.concat([s10, s11], axis=1, join="inner").dropna()
    both.columns = ["v10", "v11"]
    lines = ["阶段4+5 合入基线对照 (冻结 v1.0.0 vs v1.1.0)", "=" * 100]
    lines.append(f"{'指标':<10}{'v1.0.0(基线)':>16}{'v1.1.0(+过滤+集中度)':>22}{'差异':>12}")
    for k, lab in [("final", "终值"), ("ann", "年化"), ("sharpe", "Sharpe"),
                   ("mdd", "MaxDD"), ("calmar", "卡玛")]:
        lines.append(f"{lab:<10}{m10[k]:>16.4f}{m11[k]:>22.4f}{m11[k]-m10[k]:>+12.4f}")
    lines.append("")
    lines.append("摩擦计数 (全段累计):")
    for k in ("n_missing", "n_trad_removed", "n_buy_block", "n_st_block", "n_suspend"):
        lines.append(f"  {k:<16} v1.0 {st10.get(k, 0):>5} | v1.1 {st11.get(k, 0):>5}")
    # 剔除原因分布
    rem_cnt = {}
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if picks is None:
            continue
        _, removed = tf5(rb, picks)
        for r in removed.values():
            rem_cnt[r] = rem_cnt.get(r, 0) + 1
    lines.append(f"  v1.1 订单名单剔除 {sum(rem_cnt.values())} 只次: "
                 + ", ".join(f"{k}={v}" for k, v in sorted(rem_cnt.items())))
    # 月频差异
    # 月频差异 (NAV 索引为 'YYYYMMDD' 字符串, 按前6位分组)
    m_last = lambda s: s.groupby(s.index.str[:6]).last()
    r10 = m_last(both["v10"]).pct_change()
    r11 = m_last(both["v11"]).pct_change()
    mdiff = (r11 - r10).dropna()
    lines.append("")
    lines.append(f"月频收益差 v1.1-v1.0: 均值 {mdiff.mean():+.2%} | 最大 {mdiff.max():+.2%} | "
                 f"最小 {mdiff.min():+.2%} | 同号率 {np.mean((mdiff >= 0) == (r11.dropna() >= r10.dropna())):.0%}")
    lines.append("  (差异最大 5 个月: " + ", ".join(
        f"{t} {d:+.2%}" for t, d in mdiff.abs().nlargest(5).items()) + ")")
    print("\n".join(lines))

    fp = os.path.join(C.OUT_DIR, "v11_compare.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "v11_compare.json"), "w", encoding="utf-8") as f:
        json.dump(dict(v10=dict(metrics=m10, **{k: int(v) for k, v in st10.items()}),
                       v11=dict(metrics=m11, **{k: int(v) for k, v in st11.items()}),
                       removed=rem_cnt), f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {fp}")


if __name__ == "__main__":
    main()
