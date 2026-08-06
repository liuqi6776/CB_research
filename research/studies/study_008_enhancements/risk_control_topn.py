# -*- coding: utf-8 -*-
"""TOP_N 扫描: 中证1000 内 Base+VAL 打分取前 k 只 + HRP 权重 + RS12 (无 MA20/DD)
对比 k ∈ {5,10,20,30,40,50} 全段 2020-2026 + OOS段 2023-2026 (与 baseline_cmp 同口径)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E

KS = [55, 60, 70, 80, 100]


def build_picks(env, k):
    """与 common.Env._build_picks 同逻辑, 仅 Top 数参数化"""
    picks_map = {}
    for rb in env.rebal:
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        fvals = {}
        for code in members:
            f1, f2, ft = env.ret_1m.get(code), env.ivol.get(code), env.turn.get(code)
            fr = env.fwd.get(code)
            if fr is None or rb not in fr.index:
                continue
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            for name in env.panels:
                p = env.panels[name].get(rb)
                if p is not None and code in p.index:
                    v = p.loc[code]
                    if np.isfinite(v):
                        row[name] = v
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < k:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = [c for c in sf.BASE_COLS + ["VAL"] if c in zdf.columns]
        has = zdf[cols].dropna()
        if len(has) < k:
            continue
        picks_map[rb] = has.mean(axis=1).nlargest(k).index.tolist()
    return picks_map


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)

    # 中证1000 指数 OOS 段基准
    idx = pd.read_parquet(os.path.join(rv.IDX_DIR, "000852.SH.parquet"))
    ix = idx.set_index("trade_date")["close"].astype(float).sort_index()
    ix.index = ix.index.astype(str)
    ix_nav = ix / ix.iloc[0]
    oos0 = "20230101"
    ix_oos = ix_nav[ix_nav.index >= oos0]
    idx_oos_cum = ix_oos.iloc[-1] / ix_oos.iloc[0] - 1.0

    rows = []
    for k in KS:
        env.picks_map = build_picks(env, k)
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=True, use_ma20=False)
        full = E.daily_stats(s)
        s_oos = s[s.index >= oos0]
        oos = E.daily_stats(s_oos)
        oos_cum = s_oos.iloc[-1] / s_oos.iloc[0] - 1.0
        rows.append(dict(k=k, cum=float(s.iloc[-1] - 1), cagr=full["cagr"], mdd=full["dd"],
                         shp=full["shp"], k_ratio=full["k"], oos_cum=float(oos_cum),
                         oos_cagr=oos["cagr"], oos_mdd=oos["dd"], oos_shp=oos["shp"],
                         oos_k=oos["k"], n_skip=sum(1 for v in env.picks_map.values() if not v)))
        print(f"[k={k:>2}] 全段 年化 {full['cagr']*100:5.2f}% MaxDD {full['dd']*100:5.2f}% "
              f"Sharpe {full['shp']:4.2f} 累计 {s.iloc[-1]-1:+6.1%} | "
              f"OOS 累计 {oos_cum:+6.1%} 年化 {oos['cagr']*100:5.2f}% MaxDD {oos['dd']*100:5.2f}% "
              f"Sharpe {oos['shp']:4.2f}", flush=True)

    print(f"\n中证1000 OOS段 累计 {idx_oos_cum:+.1%}")
    fp = os.path.join(C.OUT_DIR, "risk_control_topn.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"[saved] {fp}")


if __name__ == "__main__":
    main()
