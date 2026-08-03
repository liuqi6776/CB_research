# -*- coding: utf-8 -*-
"""debug: MA20 三档月度明细 (定位 MaxDD 时段与仓位行为)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic import style_factors as sf

COST = rv.COST_BPS / 10000.0
TOP_N = rv.TOP_N
MA20_DEEP = 0.97


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

    nav_rb = {rebal[0]: 1.0}
    print(f"\n{'月':>9}{'RS12':>6}{'组合月收益':>10}{'w均值':>7}{'NAV':>9}{'回撤':>8}")
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        rb_next = rebal[i + 1]
        if rb not in picks_map:
            nav_rb[rb_next] = nav_rb.get(rb, 1.0)
            continue
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        comb = pct_df.reindex(columns=picks_map[rb]).reindex(hold).fillna(0.0) / 100.0
        comb_ret = comb.mean(axis=1)
        e_ret = etf_ret.reindex(hold).fillna(0.0)
        rs12_on = bool(sig_rs12.loc[rb]) if rb in sig_rs12.index else True
        nav = nav_rb.get(rb, 1.0)
        hwm = max(nav, max(nav_rb.values()) if nav_rb else nav)
        ws = []
        for t in hold:
            r_t = e_ret.loc[t]
            if rs12_on:
                w = 1.0
                c, m = idx_close_1.get(t, np.nan), ma20_1.get(t, np.nan)
                if np.isfinite(c) and np.isfinite(m):
                    w = 1.0 if c >= m else (0.5 if c >= MA20_DEEP * m else 0.0)
                ws.append(w)
                r_t = w * comb_ret.loc[t]
            nav *= (1.0 + r_t)
            hwm = max(hwm, nav)
        nav *= (1.0 - COST)
        nav_rb[rb_next] = nav
        dd = nav / hwm - 1.0
        cw = comb_ret.mean() if rs12_on else 0.0
        print(f"{rb:<9}{'强' if rs12_on else '弱':>5}{cw:>10.2%}{np.mean(ws) if ws else 1:>7.0%}"
              f"{nav:>9.3f}{dd:>8.2%}", flush=True)


if __name__ == "__main__":
    main()
