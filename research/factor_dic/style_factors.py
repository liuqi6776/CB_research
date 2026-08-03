# -*- coding: utf-8 -*-
"""
风格因子验证: Buffett 选股 / 价值系(EP,BP,SP,DP) / 质量系(ROE,毛利率,低负债) /
成长系(利润增速,营收增速) / GARP(EP+成长) —— 对主策略(ENH 3因子)的增量贡献

统一框架(与 21 因子/合并回测可比):
  - 样本: 2020.01~2026.06, 中证1000成分股, 月末调仓, Top50, 20bps
  - PIT: 估值取调仓日 daily_basic; 财务取 ann_date<=调仓日 最新 fina_indicator
  - 对比: 单因子 / BASE(3因子) / BASE+因子 / +RS12择时(vs ETF512100)

输出: results/style_factors.txt
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic import lynch_factor as lf

OUT_DIR = rv.OUT_DIR
COST = rv.COST_BPS / 10000.0
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FINA_ALL = os.path.join(BASE_DIR, "fina_all.parquet")
TOP_N = rv.TOP_N
BASE_COLS = ["ret_1m", "ivol", "turn"]


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def load_valuation(rebal_dates, all_codes):
    """调仓日估值面板: {rb: DataFrame(ts_code index, pe_ttm,pb,ps_ttm,dv_ttm,...)}"""
    out = {}
    for rb in rebal_dates:
        fp = os.path.join(lf.PE_DIR, f"{rb}.parquet")
        if not os.path.exists(fp):
            continue
        df = pd.read_parquet(fp).dropna(subset=["pe_ttm", "pb", "ps_ttm", "dv_ttm"], how="all")
        df = df[df["ts_code"].astype(str).isin(all_codes)]
        if not df.empty:
            df = df.set_index("ts_code")
            out[rb] = df
    return out


def build_funda_pit(rebal_dates, all_codes):
    """财务字段 PIT 面板: {rb: DataFrame(index=ts_code)}"""
    if not os.path.exists(FINA_ALL):
        return {}
    fina = pd.read_parquet(FINA_ALL)
    fina["ann_date"] = fina["ann_date"].astype(str).str[:8]
    fina = fina.sort_values("ann_date")
    out = {}
    for rb in rebal_dates:
        latest = fina[fina["ann_date"] <= rb].drop_duplicates("ts_code", keep="last")
        if not latest.empty:
            latest = latest.set_index("ts_code")
            out[rb] = latest[latest.index.isin(all_codes)]
    return out


def build_factors(val_map, funda_map, rebal_dates):
    """构造各风格因子月度面板: {name: {rb: Series(index=ts_code, 高=好)}}"""
    panels = {k: {} for k in ["EP", "BP", "SP", "DP", "ROE", "GM", "LEV", "GROW", "SGROW"]}
    for rb in rebal_dates:
        val = val_map.get(rb)
        fun = funda_map.get(rb)
        if val is None:
            continue
        pe, pb = val["pe_ttm"], val["pb"]
        ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)
        bp = (1.0 / pb).replace([np.inf, -np.inf], np.nan)
        panels["EP"][rb] = ep[ep > 0]
        panels["BP"][rb] = bp[bp > 0]
        if "ps_ttm" in val.columns:
            sp = (1.0 / val["ps_ttm"]).replace([np.inf, -np.inf], np.nan)
            panels["SP"][rb] = sp[sp > 0]
        if "dv_ttm" in val.columns:
            panels["DP"][rb] = val["dv_ttm"]
        if fun is not None:
            for f, c in [("ROE", "roe"), ("GM", "grossprofit_margin"),
                         ("LEV", "debt_to_assets"), ("GROW", "netprofit_yoy"),
                         ("SGROW", "q_sales_yoy")]:
                if c in fun.columns:
                    s = fun[c]
                    if f == "LEV":
                        s = -s                       # 低负债 = 好
                    panels[f][rb] = s
    # 合成价值因子 VAL = BP+SP+DP 平均 z-score (低估值=好)
    for rb in rebal_dates:
        comps = []
        for k in ["BP", "SP", "DP"]:
            s = panels[k].get(rb)
            if s is not None and len(s) > 20:
                comps.append((s - s.mean()) / (s.std() + 1e-8))
        if len(comps) >= 2:
            m = pd.concat(comps, axis=1).mean(axis=1).dropna()
            panels.setdefault("VAL", {})[rb] = m
    return panels


def winsorize_series(s):
    q1, q99 = s.quantile(0.01), s.quantile(0.99)
    return s.clip(q1, q99)


def main():
    t0 = time.time()
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]
    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"[load] 调仓日 {len(rebal)}, 成分股 {len(all_codes)}", flush=True)

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    ind_map = cb.load_industry_map()
    val_map = load_valuation(rebal, all_codes)
    funda_map = build_funda_pit(rebal, all_codes)
    panels = build_factors(val_map, funda_map, rebal)
    print(f"[load] 估值面板 {len(val_map)} 月, 财务面板 {len(funda_map)} 月, "
          f"耗时 {time.time()-t0:.0f}s", flush=True)

    sml = load_idx("000852.SH")
    big = load_idx("000300.SH")
    etf = load_idx("512100.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    # ---------- 单因子验证 ----------
    print("\n" + "=" * 100)
    print("单因子验证 (月度截面 Rank IC, 未来20日)")
    print("=" * 100)
    ic_stats = {}
    fwd_series = {c: fwd[c] for c in fwd if c in all_codes}
    for name, panel in panels.items():
        ic_list = []
        for rb in rebal:
            s = panel.get(rb)
            if s is None or len(s) < 50:
                continue
            members = rv.load_index_weight(rb)
            if members is None:
                continue
            s = s.reindex([c for c in s.index if c in members]).dropna()
            if len(s) < 50:
                continue
            fr = {}
            for c in s.index:
                if c in fwd_series and rb in fwd_series[c].index:
                    fr[c] = fwd_series[c].loc[rb]
            r = pd.Series(fr)
            s = s.reindex(r.index).dropna()
            r = r.reindex(s.index).dropna()
            if len(s) < 50:
                continue
            fw = winsorize_series(s)
            ic = fw.rank().corr(r.rank())
            if np.isfinite(ic):
                ic_list.append(ic)
        if not ic_list:
            continue
        ics = pd.Series(ic_list)
        t_val, _ = rv.newey_west_t(ics.values)
        ic_stats[name] = (ics.mean(), ics.mean() / ics.std(ddof=1), t_val, (ics > 0).mean(), len(ics))
        print(f"  {name:<8} IC={ics.mean():.4f}  ICIR={ic_stats[name][1]:.3f}  "
              f"NWt={t_val:.2f}  正IC={(ics>0).mean():.1%}  n={len(ics)}", flush=True)

    # ---------- 合并回测 ----------
    print("\n" + "=" * 100)
    print("合并回测: BASE(3因子) vs BASE+X, 全部叠加 RS12 择时(vs ETF512100)")
    print("=" * 100)
    results = []
    port = {}
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
        zdf = fdf.apply(winsorize_series).apply(lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        ind = pd.Series({c: ind_map.get(c, "NA") for c in fdf.index}, index=fdf.index)

        def _run(name, cols, neut=False):
            if not all(c in zdf.columns for c in cols):
                return
            has = zdf[cols].dropna()
            if len(has) < TOP_N:
                return
            sc = has.mean(axis=1)
            if neut:
                sc = sc.groupby(ind).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
            picks = sc.nlargest(TOP_N).index
            sub = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
            port.setdefault(name, []).append(dict(rb=rb, rb_next=rb_next, net=(1 + sub.mean(axis=1)).prod() - 1 - COST))

        _run("BASE", BASE_COLS)
        for name in panels:
            _run(f"BASE+{name}", BASE_COLS + [name])
            _run(f"{name}", [name])

    def _bench_rows(rows):
        bm = {}
        for r in rows:
            hi, hn = trade_dates.index(r["rb"]), trade_dates.index(r["rb_next"])
            hold = trade_dates[hi + 1:hn + 1]
            b = sml["pct_chg"].reindex(hold).fillna(0.0) / 100.0
            bm[r["rb"]] = (1 + b).prod() - 1
            be = etf["pct_chg"].reindex(hold).fillna(0.0) / 100.0
            bm_e[r["rb"]] = (1 + be).prod() - 1
        return bm

    bm_e = {}
    bm_i = pd.Series(_bench_rows(port.get("BASE", [])))
    bm_e = pd.Series(bm_e)

    def stats(pr, bm):
        pr, bm = pd.Series(pr), pd.Series(bm)
        navs = (1 + pr).cumprod()
        years = len(pr) / 12.0
        return dict(cagr=navs.iloc[-1] ** (1 / years) - 1,
                    sharpe=pr.mean() / pr.std(ddof=1) * np.sqrt(12),
                    mdd=(navs.cummax() - navs).max(), win=(pr > 0).mean(),
                    excess=(1 + pr).prod() / (1 + bm).prod() - 1, n=len(pr))

    print(f"\n{'策略':<22}{'n':>4}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'月胜率':>8}{'超额v000852':>12}{'超额vETF':>10}")
    summary = {}
    for name in ["BASE"] + [f"BASE+{n}" for n in panels] + list(panels):
        pr = pd.Series({r["rb"]: r["net"] for r in port.get(name, [])})
        if pr.empty:
            continue
        s0 = stats(pr, bm_i.reindex(pr.index))
        s0e = stats(pr, bm_e.reindex(pr.index))
        pr_t = pr.where(sig_rs12.reindex(pr.index), bm_e.reindex(pr.index))
        st = stats(pr_t, bm_e.reindex(pr.index))
        summary[name] = (s0, s0e, st)
        lbl = name + (" [单因子]" if name in panels else "")
        print(f"{lbl:<22}{s0['n']:>4}{s0['cagr']:>8.2%}{s0['sharpe']:>8.2f}{s0['mdd']:>9.2%}"
              f"{s0['win']:>8.1%}{s0['excess']:>12.2%}{s0e['excess']:>10.2%}")
        if name != "BASE" and name not in panels:
            print(f"{'  +RS12择时(vs ETF)':<22}{st['n']:>4}{st['cagr']:>8.2%}{st['sharpe']:>8.2f}"
                  f"{st['mdd']:>9.2%}{st['win']:>8.1%}{'-':>12}{st['excess']:>10.2%}")

    # BASE 自身择时
    pr_b = pd.Series({r["rb"]: r["net"] for r in port["BASE"]})
    st_b = stats(pr_b.where(sig_rs12.reindex(pr_b.index), bm_e.reindex(pr_b.index)), bm_e.reindex(pr_b.index))
    print(f"BASE+RS12择时(vs ETF)      {st_b['n']:>4}{st_b['cagr']:>8.2%}{st_b['sharpe']:>8.2f}"
          f"{st_b['mdd']:>9.2%}{st_b['win']:>8.1%}{'-':>12}{st_b['excess']:>10.2%}")

    # 保存
    with open(os.path.join(OUT_DIR, "style_factors.txt"), "w", encoding="utf-8") as fh:
        for k, v in ic_stats.items():
            fh.write(f"{k}: IC={v[0]:.4f} ICIR={v[1]:.3f} NWt={v[2]:.2f} pos={(v[3]):.3f} n={v[4]}\n")
        for k, (s0, s0e, st) in summary.items():
            fh.write(f"{k}: cagr={s0['cagr']:.4f} sharpe={s0['sharpe']:.3f} "
                     f"exvETF={s0e['excess']:.4f} exvETF_timed={st['excess']:.4f}\n")
    print(f"\n[保存] {os.path.join(OUT_DIR, 'style_factors.txt')}")


if __name__ == "__main__":
    main()
