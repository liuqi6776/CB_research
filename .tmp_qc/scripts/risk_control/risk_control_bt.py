# -*- coding: utf-8 -*-
"""
回撤控制方向统一回测 (BASE+VAL+RS12 框架, 2020-2026, 中证1000 Top50 月度调仓)
日频风控叠加在月度持有期上, 参数化变体对比:
  1. BASE+VAL         无风控 (当前最优)
  2. +MA20三档        000852 vs MA20 -> 1.0/0.5/0.0 (deep=0.97)
  3. +MA20三档098     同上, 深破位阈值 0.98 (更早降仓)
  4. +VolTarget20     仓位=clip(20%σ/指数σ20, 0.2, 1.0)
  5. +VolTarget15     仓位=clip(15%σ/指数σ20, 0.2, 1.0)
  6. +DD触发          高水位回撤 -15%半仓/-25%空仓/-5%修复
  7. +DD触发1018      更激进: -10%半仓/-18%空仓/-3%修复
  8. +CPPI(TIPP)      floor=max(floor,0.90*hwm), w=min(3*(A-F)/A,1)
  9. +CPPI085         m=2.5, α=0.85 (更早锁定, 更慢加仓)
风控降仓部分按现金(0收益)缓冲; RS12 弱时持 512100 ETF 不变; 月度调仓成本 20bps。
MA20/Vol 仓位信号均取 T-1 日收盘已知信息、T 日生效 (2026-08-03 修复同日前视)。
输出: results/risk_control_bt.txt, results/risk_control_nav.png
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic import style_factors as sf

OUT_DIR = rv.OUT_DIR
COST = rv.COST_BPS / 10000.0
TOP_N = rv.TOP_N
SQRT_242 = np.sqrt(242.0)

# (label, 风控类型, 参数)  类型: None/ma20/vol/dd/cppi
VARIANTS = [
    ("BASE+VAL", None, {}),
    ("+MA20三档", "ma20", {"deep": 0.97}),
    ("+MA20三档098", "ma20", {"deep": 0.98}),
    ("+VolTarget20", "vol", {"tgt": 0.20, "floor_w": 0.20}),
    ("+VolTarget15", "vol", {"tgt": 0.15, "floor_w": 0.20}),
    ("+DD触发", "dd", {"half": -0.15, "zero": -0.25, "fix": -0.05}),
    ("+DD触发1018", "dd", {"half": -0.10, "zero": -0.18, "fix": -0.03}),
    ("+CPPI(TIPP)", "cppi", {"m": 3.0, "alpha": 0.90}),
    ("+CPPI085", "cppi", {"m": 2.5, "alpha": 0.85}),
]


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


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

    idx_ret = sml["pct_chg"] / 100.0
    idx_close = sml["close"]
    ma20 = idx_close.rolling(20).mean()
    idx_vol20 = idx_ret.rolling(20).std() * SQRT_242
    # 修复同日信号前视: 仓位信号一律取 T-1 日收盘已知信息, T 日生效
    idx_close_1 = idx_close.shift(1)
    ma20_1 = ma20.shift(1)
    idx_vol20_1 = idx_vol20.shift(1)
    etf_ret = etf["pct_chg"] / 100.0

    # ---------- 每月 Top50 (BASE+VAL) ----------
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
    print(f"[load] Top50 月份 {len(picks_map)}, 耗时 {time.time()-t0:.0f}s", flush=True)

    # ---------- 日频风控回测 ----------
    labels = [v[0] for v in VARIANTS]
    nav_rb = {lb: {rebal[0]: 1.0} for lb in labels}
    avg_w = {lb: [] for lb in labels}          # RS12强时段的日均仓位
    # DD/CPPI/TIPP 跨期状态: 高水位/floor/半仓状态从 1.0 起跨月延续 (修复每月重置)
    st_hwm = {lb: 1.0 for lb in labels}
    st_floor = {lb: (par["alpha"] if rtype == "cppi" else 0.90)
                for (lb, rtype, par) in VARIANTS}
    st_w_half = {lb: False for lb in labels}
    # DD 变体 shadow NAV: 假想未降仓的组合净值, 用于计算回撤与恢复
    # (修复: 原用实际 NAV 计算 dd, 空仓后 NAV 冻结导致无法自行恢复到恢复线)
    st_shadow = {lb: 1.0 for lb in labels}
    st_shadow_hwm = {lb: 1.0 for lb in labels}
    rs12_days = 0

    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        rb_next = rebal[i + 1]
        if rb not in picks_map:
            for lb in labels:
                nav_rb[lb][rb_next] = nav_rb[lb].get(rb, 1.0)
            continue
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        picks = picks_map[rb]
        comb = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
        comb_ret = comb.mean(axis=1)
        e_ret = etf_ret.reindex(hold).fillna(0.0)
        rs12_on = bool(sig_rs12.loc[rb]) if rb in sig_rs12.index else True
        if rs12_on:
            rs12_days += len(hold)

        for (lb, rtype, par) in VARIANTS:
            nav = nav_rb[lb].get(rb, 1.0)
            # DD/CPPI/TIPP 跨期状态 (不每月重置, 修复): 高水位/floor/半仓状态跨月延续
            hwm = st_hwm[lb]
            floor = st_floor[lb]
            w_half = st_w_half[lb]
            shadow = st_shadow[lb]
            shadow_hwm = st_shadow_hwm[lb]
            # DD: T 日仓位由 T-1 末 shadow 回撤决定 (跨月状态重算, 无前视)
            dd_prev = shadow / shadow_hwm - 1.0
            ws = []
            for t in hold:
                r_t = e_ret.loc[t]
                if rs12_on:
                    w = 1.0
                    if rtype == "ma20":
                        c, m = idx_close_1.get(t, np.nan), ma20_1.get(t, np.nan)
                        if np.isfinite(c) and np.isfinite(m):
                            w = 1.0 if c >= m else (0.5 if c >= par["deep"] * m else 0.0)
                    elif rtype == "vol":
                        v = idx_vol20_1.get(t, np.nan)
                        if np.isfinite(v) and v > 0:
                            w = float(np.clip(par["tgt"] / v, par["floor_w"], 1.0))
                    elif rtype == "dd":
                        # 用 T-1 末的 shadow 回撤决定 T 日仓位 (修复: 先决策再吃当日收益, 无前视)
                        if w_half and dd_prev >= par["fix"]:
                            w_half = False
                        if dd_prev <= par["half"]:
                            w_half = True
                        w = 0.0 if dd_prev <= par["zero"] else (0.5 if w_half else 1.0)
                    elif rtype == "cppi":
                        floor = max(floor, par["alpha"] * hwm)
                        w = float(np.clip(par["m"] * (nav - floor) / nav, 0.0, 1.0)) if nav > 0 else 0.0
                    ws.append(w)
                    r_t = w * comb_ret.loc[t]      # 降仓部分按现金缓冲
                nav *= (1.0 + r_t)
                hwm = max(hwm, nav)
                # shadow NAV: 假想始终满仓股票组合, 与 RS12 状态无关 (修复: 弱段也更新)
                # 更新发生在当日收益应用之后, 供 T+1 日决策使用 (无前视)
                if rtype == "dd":
                    shadow *= (1.0 + comb_ret.loc[t])
                    shadow_hwm = max(shadow_hwm, shadow)
                    dd_prev = shadow / shadow_hwm - 1.0
            if ws:
                avg_w[lb].append(np.mean(ws))
            nav *= (1.0 - COST)
            nav_rb[lb][rb_next] = nav
            st_hwm[lb] = hwm
            st_floor[lb] = floor
            st_w_half[lb] = w_half
            st_shadow[lb] = shadow
            st_shadow_hwm[lb] = shadow_hwm

    # ---------- 汇总 ----------
    bm_e_m, bm_i_m = {}, {}
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        rb_next = rebal[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        bm_e_m[rb_next] = (1 + etf_ret.reindex(hold).fillna(0.0)).prod() - 1
        bm_i_m[rb_next] = (1 + idx_ret.reindex(hold).fillna(0.0)).prod() - 1
    bm_e_m, bm_i_m = pd.Series(bm_e_m), pd.Series(bm_i_m)

    print("\n" + "=" * 110)
    print("回撤控制对比 (BASE+VAL + RS12 框架, 2020-2026, 月度调仓 Top50, 20bps, 风控降仓吃现金)")
    print("=" * 110)
    print(f"\n{'策略':<16}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'月胜率':>8}{'超额vETF':>10}{'卡玛':>7}{'强段均仓':>8}")
    out_lines = []
    nav_series = {}
    for lb in labels:
        s = pd.Series(nav_rb[lb]).sort_index()
        pr = s.pct_change().dropna()
        navs_c = (1 + pr).cumprod()
        years = len(pr) / 12.0
        cagr = navs_c.iloc[-1] ** (1 / years) - 1
        sharpe = pr.mean() / pr.std(ddof=1) * np.sqrt(12)
        mdd = ((navs_c.cummax() - navs_c) / navs_c.cummax()).max()
        ex = (1 + pr).prod() / (1 + bm_e_m.reindex(pr.index)).prod() - 1
        calmar = cagr / mdd if mdd > 0 else np.nan
        aw = np.mean(avg_w[lb]) if avg_w[lb] else np.nan
        line = f"{lb:<16}{cagr:>8.2%}{sharpe:>8.2f}{mdd:>9.2%}{(pr>0).mean():>8.1%}{ex:>10.2%}{calmar:>7.2f}{aw:>8.0%}"
        print(line)
        out_lines.append(line)
        nav_series[lb] = s

    def bench_stats(lbl, bm):
        bm = bm.dropna()
        navs_c = (1 + bm).cumprod()
        years = len(bm) / 12.0
        line = f"{lbl:<16}{navs_c.iloc[-1]**(1/years)-1:>8.2%}{bm.mean()/bm.std(ddof=1)*np.sqrt(12):>8.2f}"
        line += f"{((navs_c.cummax()-navs_c)/navs_c.cummax()).max():>9.2%}"
        print(line)
        out_lines.append(line)
    bench_stats("基准000852", bm_i_m)
    bench_stats("基准512100ETF", bm_e_m)

    with open(os.path.join(OUT_DIR, "risk_control_bt.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")

    # ---------- 图 ----------
    fig, ax = plt.subplots(figsize=(13, 7.5))
    colors = {"BASE+VAL": "#888", "+MA20三档": "#c33", "+MA20三档098": "#f66",
              "+VolTarget20": "#282", "+VolTarget15": "#3a6",
              "+DD触发": "#e90", "+DD触发1018": "#fa3",
              "+CPPI(TIPP)": "#a4c", "+CPPI085": "#73f"}
    x_all = sorted(nav_series[labels[0]].index)
    for lb in labels:
        s = nav_series[lb].reindex(x_all).ffill().fillna(1.0)
        ax.plot(np.arange(len(x_all)), s.values, label=lb, lw=1.6, color=colors[lb])
    ax.plot(np.arange(len(x_all)), (1 + bm_e_m.reindex(x_all).fillna(0)).cumprod().values,
            label="512100ETF", lw=1.2, ls="--", color="#999")
    ax.set_yscale("log")
    ax.set_ylabel("净值(对数)")
    ax.set_title("回撤控制对比: 9 变体 (BASE+VAL+RS12 + 风控, 2020-2026)", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(len(x_all)), 0, 1,
                     where=sig_rs12.reindex(x_all).fillna(False).values, color="#0c6", alpha=0.15)
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)
    fp = os.path.join(OUT_DIR, "risk_control_nav.png")
    plt.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[保存图] {fp}   (RS12强时段共 {rs12_days} 交易日)")


if __name__ == "__main__":
    main()
