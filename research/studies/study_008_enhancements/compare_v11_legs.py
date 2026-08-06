# -*- coding: utf-8 -*-
"""v1.1.0 防御腿对照 (引擎口径): ETF(512100) vs 短债(511260)

RS12 空头期的防御腿选择 — 在 v1.1.0 (阶段4+5 合入) 基础上比较:
  leg='etf'  : 基线, 持中证1000 ETF (名义防御实为满仓小盘)
  leg='bond' : 短债腿, 持十年国债ETF (票息+久期)
输出: results/v11_legs.txt | .json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
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
    bond = pd.read_parquet(os.path.join(C.DATA_DIR, "bond_511260.parquet")).set_index("trade_date")
    b_open_s = bond["open"].astype(float)
    b_close_s = bond["close"].astype(float)

    def _conc(rb, w, nav_pre):
        return apply_concentration(w, ind_map=ind_map, cap_stock=0.04, cap_ind=0.20,
                                   cap_top5=0.20,
                                   amount60=amount60_at(amount_df, td, rb),
                                   nav_pre=nav_pre, cap_amount=0.05, scale=1e8)

    kw = dict(use_hrp=True, use_ma20=False, st_map=st_map,
              limit_sets=(one_up, one_dn), tradable=tf5, concentration=_conc)
    s_etf, st_etf = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                   e_ovn, e_intra, **kw, leg="etf")
    s_bnd, st_bnd = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                   e_ovn, e_intra, **kw, leg="bond",
                                   bond_open_s=b_open_s, bond_close_s=b_close_s)

    m_etf, m_bnd = _metrics(s_etf), _metrics(s_bnd)
    lines = ["v1.1.0 防御腿对照 (引擎口径, 阶段4+5 已合入)", "=" * 100]
    lines.append(f"{'指标':<10}{'ETF腿(512100)':>18}{'短债腿(511260)':>18}{'差异':>12}")
    for k, lab in [("final", "终值"), ("ann", "年化"), ("sharpe", "Sharpe"),
                   ("mdd", "MaxDD"), ("calmar", "卡玛")]:
        lines.append(f"{lab:<10}{m_etf[k]:>18.4f}{m_bnd[k]:>18.4f}{m_bnd[k]-m_etf[k]:>+12.4f}")
    lines.append(f"  摩擦 v1.1: 剔除ST {st_etf['n_trad_removed']} 只次 | 阻塞 买{st_etf['n_buy_block']} "
                 f"停牌{st_etf['n_suspend']} | n_missing {st_etf['n_missing']} 月 (两腿一致)")
    # 逐年收益
    ml = lambda s: s.groupby(s.index.str[:6]).last()
    y10, y11 = ml(s_etf), ml(s_bnd)
    years = sorted(set(k[:4] for k in y10.index))
    lines.append("")
    lines.append("逐年收益:")
    lines.append(f"{'年份':<8}{'ETF腿':>12}{'短债腿':>12}{'差异':>12}")
    for y in years:
        r_etf = y10.loc[[k for k in y10.index if k.startswith(y)]].pct_change().sum()
        r_bnd = y11.loc[[k for k in y11.index if k.startswith(y)]].pct_change().sum()
        lines.append(f"{y:<8}{r_etf:>12.2%}{r_bnd:>12.2%}{r_bnd-r_etf:>+12.2%}")
    # RS12 空头期月度
    lines.append("")
    lines.append("RS12 空头期 (防御腿持有) 月度收益:")
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if not len(hold) or bool(rs12_on):
            continue
        h = [t for t in hold if t in s_etf.index]
        if len(h) < 2:
            continue
        r_etf = s_etf.loc[h[-1]] / s_etf.loc[h[0]] - 1.0
        r_bnd = s_bnd.loc[h[-1]] / s_bnd.loc[h[0]] - 1.0
        lines.append(f"  {rb}  ETF {r_etf:+.2%} | 短债 {r_bnd:+.2%} | 差 {r_bnd-r_etf:+.2%}")
    print("\n".join(lines))
    fp = os.path.join(C.OUT_DIR, "v11_legs.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "v11_legs.json"), "w", encoding="utf-8") as f:
        json.dump(dict(etf=dict(metrics=m_etf, **{k: int(v) for k, v in st_etf.items()}),
                       bond=dict(metrics=m_bnd, **{k: int(v) for k, v in st_bnd.items()})),
                  f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {fp}")


if __name__ == "__main__":
    main()
