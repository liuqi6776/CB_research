# -*- coding: utf-8 -*-
"""阶段6 (用户验收 RS12 分离): 三腿并行对照 — RS12 空头防御腿选择

基线冻结口径 (v1.0.0): RS12>0 → 中证1000 Top60 股票腿; RS12≤0 → 512100 (中证1000 ETF).
本脚本对照 RS12 空头期的三种防御腿:
  A. ETF 腿 (基线, 512100 中证1000 ETF)     — 暴露小盘风险, 动量回落时吃指数跌幅
  B. 现金腿 (空仓持币, 0 收益)              — 完全规避权益暴露
  C. 短债腿 (511260 十年国债ETF)            — 票息+久期收益, 对冲权益回撤

口径与阶段4/5 生产路径一致 (tradable + concentration), 保证与阶段5 结果可比.
输出: results/rs12_legs.txt | .json
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
from research.studies.study_008_enhancements import ledger as L
from research.studies.study_008_enhancements.concentration import apply_concentration, amount60_at
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df

BOND_CODE = "511260.SH"          # 十年国债ETF (中债-7-10年国债财富指数)
BOND_START = "20200101"


def ensure_bond_data():
    """下载/缓存 511260 日线到 data/bond_511260.parquet (一次性)"""
    fp = os.path.join(C.DATA_DIR, "bond_511260.parquet")
    if os.path.exists(fp):
        return fp
    import tushare as ts
    from config.settings import settings
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    df = pro.fund_daily(ts_code=BOND_CODE, start_date=BOND_START,
                        end_date="20260804")
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date")
    df.to_parquet(fp, index=False)
    print(f"[bond] {BOND_CODE} 下载完成 {len(df)} 行 -> {fp}")
    return fp


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)
    etf = C.load_idx("512100.SH")
    e_open_s = etf["open"].astype(float)
    e_close_s = etf["close"].astype(float)

    # 债券腿数据 (511260 十年国债ETF)
    bond = pd.read_parquet(ensure_bond_data()).set_index("trade_date")
    b_open_s = bond["open"].astype(float)
    b_close_s = bond["close"].astype(float)

    # 生产路径 (阶段4+5): 可交易过滤 + 集中度约束
    st_map = E.load_st_intervals()
    one_up, one_dn = E.build_limit_sets(open_df, high_df, low_df, pct_df, env.all_codes)
    amount_df = load_amount_df(env, td)
    tf5 = Tradability(td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                      st_map=st_map, min_vol=12.0, pct_df=pct_df)
    ind_map = C.load_industry_map()

    def _conc(rb, w, nav_pre):
        return apply_concentration(w, ind_map=ind_map, cap_stock=0.04, cap_ind=0.20,
                                   cap_top5=0.20,
                                   amount60=amount60_at(amount_df, td, rb),
                                   nav_pre=nav_pre, cap_amount=0.05, scale=1e8)

    def run(leg):
        s, st = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s,
                             st_map=st_map, one_up=one_up, one_dn=one_dn,
                             tradable=tf5, concentration=_conc, leg=leg,
                             bond_open_s=b_open_s, bond_close_s=b_close_s)
        return s, st

    rows = []
    navs = {}
    for label, leg in [("ETF腿(基线 512100)", "etf"), ("现金腿(持币)", "cash"),
                       ("短债腿(511260)", "bond")]:
        s, st = run(leg)
        navs[leg] = s
        m = C.monthly_metrics(s)
        rows.append((label, s, dict(te=f"{st['avg_te']*100:.2f}%", etf=f"{st['final_etf']:.3f}",
                                    cash=f"{st['final_cash']:.4f}")))
        print(f"[{label}] 年化 {m['ann']*100:.2f}% | Sharpe {m['sharpe']:.2f} | "
              f"MaxDD {m['mdd']*100:.2f}% | 卡玛 {m['calmar']:.2f} | 终值 {m['final']:.4f}")

    # RS12 空头期逐月收益 (防御腿差异的主要来源)
    lines = ["阶段6: RS12 三腿并行对照 (RS12≤0 防御腿: ETF / 现金 / 短债)", "=" * 90]
    table, out = C.metrics_table(rows)
    lines.append(table)
    lines.append("")
    lines.append("RS12 空头期 (防御腿持有期) 逐月收益对照:")
    segs = list(env.month_segments())
    for i, (rb, rb_next, hold, picks, comb, e_ret, rs12_on) in enumerate(segs):
        if not len(hold) or bool(rs12_on):
            continue
        parts = [f"{rb} "]
        for leg in ("etf", "cash", "bond"):
            nav = navs[leg]
            h = [t for t in hold if t in nav.index]
            if len(h) >= 2:
                r = nav.loc[h[-1]] / nav.loc[h[0]] - 1.0
                parts.append(f"{leg} {r:+.2%}")
        lines.append("  " + " ".join(parts))
        print("  " + " ".join(parts))

    fp = os.path.join(C.OUT_DIR, "rs12_legs.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "rs12_legs.json"), "w", encoding="utf-8") as f:
        json.dump({leg: {k: float(v) for k, v in C.monthly_metrics(navs[leg]).items()}
                   for leg in navs}, f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {fp}")


if __name__ == "__main__":
    main()
