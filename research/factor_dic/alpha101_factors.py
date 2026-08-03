# -*- coding: utf-8 -*-
"""
WorldQuant Alpha101 代表因子第二层验证 (2020-2026, 中证1000, 月频调仓 Top50)
- 12 个可计算、覆盖不同 alpha 源的 Alpha101 因子, 在调仓日采样日频信号
- 验证: 单因子 IC / 与主策略 BASE(3因子) 截面相关 / 并入 BASE 的增量(+RS12择时)
- 简化实现说明: 保留 alpha 直觉; ts_rank 用 rolling.rank(pct=True); 方向统一高=好

输出: results/alpha101_factors.txt
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb

OUT_DIR = rv.OUT_DIR
COST = rv.COST_BPS / 10000.0
BASE_COLS = ["ret_1m", "ivol", "turn"]


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def build_ohlcv_panels(trade_dates, all_codes):
    """日频宽表面板: 每个字段一个 DataFrame(date x code)"""
    frames = []
    for d in trade_dates:
        fp = os.path.join(rv.DAY_DIR, f"{d}.parquet")
        if not os.path.exists(fp):
            continue
        df = rv._read_daily_parquet(fp, d)
        if df is None:
            continue
        df = df[df["ts_code"].astype(str).isin(all_codes)]
        if df.empty:
            continue
        df["_d"] = d
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big["vwap"] = (big["amount"] / (big["vol"] * 100 + 1e-8)).fillna(big["close"])
    out = {}
    for f in ["open", "high", "low", "close", "vol", "amount", "vwap"]:
        out[f] = big.pivot_table(index="_d", columns="ts_code", values=f)
    out["ret"] = out["close"].pct_change()
    out["ret"].iloc[0] = 0.0
    del big
    return out


def build_alpha101(p, rebal):
    """计算 Alpha101 因子宽表(date x code), 返回 {name: DataFrame}"""
    close, vol, ret = p["close"], p["vol"], p["ret"]
    open_, high, low, vwap = p["open"], p["high"], p["low"], p["vwap"]
    f = {}

    # A1: 动量-波动择时 (过去5日中信号最强位置) -> 高=好
    std20 = ret.rolling(20).std()
    base = ret.where(ret < 0, close)          # ret<0 用 stddev(ret,20), 否则 close
    base = base.where(ret < 0, close)
    sig = base.where(ret < 0, std20)
    pw = np.sign(sig) * sig ** 2
    f["A1"] = pw.rolling(5).apply(np.argmax, raw=True).rank(axis=0) / pw.count(axis=0)
    # A4: 低位 + 低量 -> 取负(低=好)
    f["A4"] = -(low.rolling(9).rank(pct=True).rank(axis=0) * vol.rolling(9).rank(pct=True).rank(axis=0))
    # A6: 反转量能 (方向反转 * 5日量) -> 取负(反转=好)
    sgn = np.sign(close - close.shift(1))
    f["A6"] = -(sgn * vol.rolling(5).sum()).rank(axis=0)
    # A12: 量变与价变反向 (放量下跌=好)
    f["A12"] = np.sign(vol.diff(1)) * (-(close.diff(1)))
    # A18: 量价相关
    c1 = close.rolling(10).corr(vol)
    c2 = vol.rolling(10).corr(close.rolling(5).rank(pct=True))
    f["A18"] = c1.rank(axis=0) * c2.rank(axis=0)
    # A31: 20日价距 + 日内实体 (简化)
    f["A31"] = ((close - close.rolling(20).mean()).rank(axis=0)
                + (close - open_).rank(axis=0))
    # A53: 日内位置 9日变化 (取负: 位置下移=好)
    pos = ((close - low) - (high - close)) / (close - low).replace(0, np.nan)
    f["A53"] = -pos.diff(9)
    # A54: 日内振幅权重 (取负)
    f["A54"] = -((low - close) * open_ ** 5) / ((low - high) * close ** 5 + 1e-8)
    # A101: 日内影线 (收盘-开盘)/日内振幅
    f["A101"] = (close - open_) / ((high - low) + 0.001)
    # A102: 上下影线差 (简化): 上影 - 下影, 取负(上影少=好)
    up_shadow = high - np.maximum(open_, close)
    dn_shadow = np.minimum(open_, close) - low
    f["A102"] = -(up_shadow - dn_shadow).rank(axis=0)
    # C20: 20日量价相关 (长周期, 高=好)
    f["C20"] = close.rolling(20).corr(vol).rank(axis=0)
    # ILLIQ: Amihud 非流动性 (|ret|/amount 20日均, 取负: 流动性高=好)
    amihud = (ret.abs() / (p["amount"] + 1e-8)).rolling(20).mean()
    f["ILLIQ"] = -amihud.rank(axis=0)

    return {k: v.reindex(rebal) for k, v in f.items()}


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
    p = build_ohlcv_panels(trade_dates, all_codes)
    alpha = build_alpha101(p, rebal)
    print(f"[load] Alpha101 面板 {len(alpha)} 个, 耗时 {time.time()-t0:.0f}s", flush=True)
    del p

    sml = load_idx("000852.SH")
    big = load_idx("000300.SH")
    etf = load_idx("512100.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    # ---------- 单因子 IC + 与 BASE 因子相关 ----------
    print("\n" + "=" * 100)
    print("单因子验证 (月度截面 Rank IC, 未来20日) + 与BASE因子截面相关")
    print("=" * 100)
    ic_stats, corr_stats = {}, {}
    fwd_series = {c: fwd[c] for c in fwd if c in all_codes}
    base_series = {"ret_1m": ret_1m, "ivol": ivol, "turn": turn}
    for name, panel in alpha.items():
        ic_list, corr_list = [], []
        for rb in rebal:
            s = panel.loc[rb] if rb in panel.index else None
            if s is None:
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
                row = {}
                for bn, bs in base_series.items():
                    bser = pd.Series({c: bs[c].loc[rb] for c in s.index
                                      if bs.get(c) is not None and rb in bs[c].index})
                    if len(bser) > 30:
                        row[bn] = fw.rank().corr(bser.reindex(fw.index).rank())
                if row:
                    corr_list.append(row)
        if not ic_list:
            continue
        ics = pd.Series(ic_list)
        t_val, _ = rv.newey_west_t(ics.values)
        ic_stats[name] = (ics.mean(), ics.mean() / ics.std(ddof=1), t_val, (ics > 0).mean(), len(ics))
        cc = pd.DataFrame(corr_list).mean()
        corr_stats[name] = cc
        print(f"  {name:<7} IC={ics.mean():.4f}  ICIR={ic_stats[name][1]:.3f}  NWt={t_val:.2f}  "
              f"正IC={(ics>0).mean():.1%}  n={len(ics)}  "
              f"ρ(ret_1m)={cc.get('ret_1m', np.nan):+.2f} ρ(ivol)={cc.get('ivol', np.nan):+.2f} "
              f"ρ(turn)={cc.get('turn', np.nan):+.2f}", flush=True)

    # ---------- 并入 BASE 回测 ----------
    print("\n" + "=" * 100)
    print("合并回测: BASE vs BASE+Alpha101, 叠加 RS12 择时(vs ETF512100)")
    print("=" * 100)
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
            for name, panel in alpha.items():
                v = panel.loc[rb, code] if rb in panel.index and code in panel.columns else np.nan
                if np.isfinite(v):
                    row[name] = v
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < rv.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        ind = pd.Series({c: ind_map.get(c, "NA") for c in fdf.index}, index=fdf.index)

        def _run(name, cols):
            if not all(c in zdf.columns for c in cols):
                return
            has = zdf[cols].dropna()
            if len(has) < rv.TOP_N:
                return
            sc = has.mean(axis=1)
            picks = sc.nlargest(rv.TOP_N).index
            sub = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
            port.setdefault(name, []).append(dict(rb=rb, rb_next=rb_next,
                                                  net=(1 + sub.mean(axis=1)).prod() - 1 - COST))

        _run("BASE", BASE_COLS)
        for nm in alpha:
            _run(f"BASE+{nm}", BASE_COLS + [nm])

    bm_e = {}
    for r in port.get("BASE", []):
        hi, hn = trade_dates.index(r["rb"]), trade_dates.index(r["rb_next"])
        hold = trade_dates[hi + 1:hn + 1]
        be = etf["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bm_e[r["rb"]] = (1 + be).prod() - 1
    bm_e = pd.Series(bm_e)

    def stats(pr, bm):
        pr, bm = pd.Series(pr), pd.Series(bm)
        navs = (1 + pr).cumprod()
        years = len(pr) / 12.0
        return dict(cagr=navs.iloc[-1] ** (1 / years) - 1,
                    sharpe=pr.mean() / pr.std(ddof=1) * np.sqrt(12),
                    mdd=(navs.cummax() - navs).max(), win=(pr > 0).mean(),
                    excess=(1 + pr).prod() / (1 + bm).prod() - 1, n=len(pr))

    print(f"\n{'策略':<22}{'n':>4}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'超额vETF':>10}{'+RS12超额':>10}")
    summary = {}
    for name in ["BASE"] + [f"BASE+{n}" for n in alpha]:
        pr = pd.Series({r["rb"]: r["net"] for r in port.get(name, [])})
        if pr.empty:
            continue
        s0 = stats(pr, bm_e.reindex(pr.index))
        pr_t = pr.where(sig_rs12.reindex(pr.index), bm_e.reindex(pr.index))
        st = stats(pr_t, bm_e.reindex(pr.index))
        summary[name] = (s0, st)
        print(f"{name:<22}{s0['n']:>4}{s0['cagr']:>8.2%}{s0['sharpe']:>8.2f}{s0['mdd']:>9.2%}"
              f"{s0['excess']:>10.2%}{st['excess']:>10.2%}", flush=True)

    with open(os.path.join(OUT_DIR, "alpha101_factors.txt"), "w", encoding="utf-8") as fh:
        for k, v in ic_stats.items():
            cc = corr_stats.get(k, {})
            fh.write(f"{k}: IC={v[0]:.4f} ICIR={v[1]:.3f} NWt={v[2]:.2f} pos={v[3]:.3f} n={v[4]} "
                     f"rho_ret={cc.get('ret_1m', float('nan')):.3f} "
                     f"rho_ivol={cc.get('ivol', float('nan')):.3f} "
                     f"rho_turn={cc.get('turn', float('nan')):.3f}\n")
        for k, (s0, st) in summary.items():
            fh.write(f"{k}: cagr={s0['cagr']:.4f} sharpe={s0['sharpe']:.3f} "
                     f"exvETF={s0['excess']:.4f} exvETF_timed={st['excess']:.4f}\n")
    print(f"\n[保存] {os.path.join(OUT_DIR, 'alpha101_factors.txt')}")


if __name__ == "__main__":
    main()
