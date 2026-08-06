# -*- coding: utf-8 -*-
"""P1-③ 停牌 / 涨跌停 / ST 摩擦建模 (残余简化清单)

真实口径 (+HRP +MA20五档098) 下叠加摩擦, 量化对净值的影响:
  基线  : 现口径 (调仓日停牌股不买入->现金, 一字板未建模, ST 未过滤)
  +ST   : 调仓日剔除 ST 股 (namechange 历史简称构建)
  +涨停 : 调仓日一字涨停股不可买 (剔除并权重归一化)
  +全摩擦: ST + 涨停 + 停牌(基线已有)
附加: 换出池一字跌停(卖不出) 频率统计 -> 判断卖出端建模必要性
输出: results/risk_control_friction.txt
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.risk_control_real import TIER5


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)

    st_map = E.load_st_intervals()
    print(f"[ST] {len(st_map)} 只有 ST 历史区间")

    codes = sorted(set(env.all_codes) & set(open_df.columns))
    one_up, one_dn = E.build_limit_sets(open_df, high_df, low_df, pct_df, codes)
    print(f"[涨跌停] 一字涨停日 {len(one_up)} (股,日), 一字跌停日 {len(one_dn)}")

    lines = ["P1-③ 停牌 / 涨跌停 / ST 摩擦建模 (真实口径 +HRP +MA20五档098)", "=" * 90,
             f"{'变体':<30}{'年化':>8}{'Sharpe':>8}{'日频MaxDD':>10}{'卡玛':>7}{'停牌股月均':>10}{'ST剔除':>8}{'涨停剔除':>9}"]
    rows = {}
    cases = [
        ("基线", None, None),
        ("+ST过滤", st_map, None),
        ("+涨停不可买", None, (one_up, one_dn)),
        ("+全摩擦(ST+涨停+停牌)", st_map, (one_up, one_dn)),
    ]
    for lb, st_map_, lim_ in cases:
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=True, use_ma20=True, tier=TIER5,
                               st_map=st_map_, limit_sets=lim_)
        m = E.daily_stats(s)
        rows[lb] = dict(**m, **st)
        lines.append(f"{lb:<30}{m['cagr']:>8.2%}{m['shp']:>8.2f}{m['dd']:>10.2%}{m['k']:>7.2f}"
                     f"{st['n_suspend'] / 79.0:>10.2f}{st['n_st_block']:>8d}{st['n_buy_block']:>9d}")
        print(f"  {lb:<30}{m['cagr']:>8.2%}  {m['dd']:>8.2%}  {m['k']:>6.2f}")

    # 换出池一字跌停 (卖不出) 频率统计
    lines.append("")
    lines.append("卖出端摩擦统计 (调仓换出股中一字跌停占比, 判断是否需建模):")
    n_dn = n_out = 0
    prev_picks = None
    for a, b, c, d, e_, f_, g in env.month_segments():
        if d is None:
            continue
        if prev_picks is not None:
            out = set(prev_picks) - set(d)
            t0 = c[0]
            dn = [x for x in out if (x, t0) in one_dn]
            n_dn += len(dn)
            n_out += len(out)
        prev_picks = d
    lines.append(f"  换出股 {n_out} 只, 其中调仓日一字跌停卖不出 {n_dn} 只 ({n_dn / max(n_out, 1):.2%})")
    print(f"  换出股 {n_out} 只, 一字跌停卖不出 {n_dn} ({n_dn / max(n_out, 1):.2%})")

    with open(os.path.join(C.OUT_DIR, "risk_control_friction.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "risk_control_friction.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {C.OUT_DIR}/risk_control_friction.txt")


if __name__ == "__main__":
    main()
