# -*- coding: utf-8 -*-
"""
回撤控制补充方向 (第5/6方向): 个股止损 + IM 空头对冲
在 BASE+VAL+RS12+MA20三档(0.98) [当前最好] 基础上叠加, 看增量效果:
  1. +止损8%    单票从调仓日起累计跌幅 -8% 剔除(转现金), 日频
  2. +止损12%   同上, 阈值 -12%
  3. +对冲07    IM 空头对冲 beta=0.7 (用 000852 日收益近似, 无基差)
  4. +对冲07B   同上 + 年化 9.3% 基差成本 (IM 贴水实测 2023-2026)
  5. +对冲10B   beta=1.0 + 基差
  6. +止损8+对冲07B  两者叠加
风控降仓/止损部分按现金(0收益)缓冲; RS12 弱时持 512100 ETF 不变; 月度调仓成本 20bps。

输出: results/risk_control_bt2.txt, results/risk_control_nav2.png
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
SL_COST = 0.01   # 止损执行摩擦: 触发转现金日扣 1% (滑点+卖出成本, 敏感性假设)

# (label, 止损阈值, 对冲beta, 基差年化, 是否叠加MA20三档)
VARIANTS = [
    ("BASE+VAL",            0.0, 0.0, 0.0, False),
    ("+MA20三档098",        0.0, 0.0, 0.0, True),
    ("+MA20+止损8",         0.08, 0.0, 0.0, True),
    ("+MA20+止损12",        0.12, 0.0, 0.0, True),
    ("+MA20+对冲07",        0.0, 0.7, 0.0, True),
    ("+MA20+对冲07B",       0.0, 0.7, 0.093, True),
    ("+MA20+对冲10B",       0.0, 1.0, 0.093, True),
    ("+MA20+止损8+对冲07B", 0.08, 0.7, 0.093, True),
]
MA20_DEEP = 0.98
BASIS_DAILY = 0.093 / 242.0


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
    # 修复同日信号前视: T-1 日收盘信号, T 日生效
    idx_close_1 = idx_close.shift(1)
    ma20_1 = ma20.shift(1)
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
        e_ret = etf_ret.reindex(hold).fillna(0.0)
        i_ret = idx_ret.reindex(hold).fillna(0.0)
        rs12_on = bool(sig_rs12.loc[rb]) if rb in sig_rs12.index else True

        # 个股止损面板: 单票从调仓日起累计跌幅阈值触发后转现金
        cum = (1 + comb).cumprod()

        for (lb, sl, beta, basis, use_ma20) in VARIANTS:
            nav = nav_rb[lb].get(rb, 1.0)
            if sl > 0:
                # 止损触发后转现金, 持有到下次调仓 (永久锁定, 修复"复活"问题)
                trig = (cum.shift(1, fill_value=1.0) < (1 - sl)).cummax().astype(bool)
                alive = ~trig
                comb_ = comb.where(alive, 0.0)
                comb_ret = comb_.mean(axis=1)
                # 止损执行摩擦: 首次触发日(锁存边沿)扣 SL_COST × 当日止损个股权重
                # (修复: 基于锁存 trig 的边沿, 重复跌破阈值不重复扣费; 未计跌停无法成交/隔夜跳空)
                new_trig = trig & ~trig.shift(1, fill_value=False)
                comb_ret = comb_ret - SL_COST * new_trig.astype(float).mean(axis=1)
            else:
                comb_ret = comb.mean(axis=1)
            hwm = nav
            ws = []
            basis_d = basis / 242.0
            for t in hold:
                r_t = e_ret.loc[t]
                if rs12_on:
                    w = 1.0
                    if use_ma20:
                        c, m = idx_close_1.get(t, np.nan), ma20_1.get(t, np.nan)
                        if np.isfinite(c) and np.isfinite(m):
                            w = 1.0 if c >= m else (0.5 if c >= MA20_DEEP * m else 0.0)
                    ws.append(w)
                    # 组合收益(MA20仓位×止损后收益) + 对冲腿(空头beta×指数 - 基差成本)
                    r_t = w * comb_ret.loc[t] - beta * w * i_ret.loc[t] - basis_d * w
                nav *= (1.0 + r_t)
                hwm = max(hwm, nav)
            if ws:
                avg_w[lb].append(np.mean(ws))
            nav *= (1.0 - COST)
            nav_rb[lb][rb_next] = nav

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

    print("\n" + "=" * 118)
    print("回撤控制补充: 个股止损 / IM空头对冲 (BASE+VAL+RS12, 月度调仓 Top50, 20bps, 2020-2026)")
    print("=" * 118)
    print(f"\n{'策略':<20}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'月胜率':>8}{'超额vETF':>10}{'卡玛':>7}{'强段均仓':>8}")
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
        line = f"{lb:<20}{cagr:>8.2%}{sharpe:>8.2f}{mdd:>9.2%}{(pr>0).mean():>8.1%}{ex:>10.2%}{calmar:>7.2f}{aw:>8.0%}"
        print(line)
        out_lines.append(line)
        nav_series[lb] = s

    def bench_stats(lbl, bm):
        bm = bm.dropna()
        navs_c = (1 + bm).cumprod()
        years = len(bm) / 12.0
        line = f"{lbl:<20}{navs_c.iloc[-1]**(1/years)-1:>8.2%}{bm.mean()/bm.std(ddof=1)*np.sqrt(12):>8.2f}"
        line += f"{((navs_c.cummax()-navs_c)/navs_c.cummax()).max():>9.2%}"
        print(line)
        out_lines.append(line)
    bench_stats("基准000852", bm_i_m)
    bench_stats("基准512100ETF", bm_e_m)

    with open(os.path.join(OUT_DIR, "risk_control_bt2.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")

    # ---------- 图 ----------
    fig, ax = plt.subplots(figsize=(13, 7.5))
    colors = {"BASE+VAL": "#888", "+MA20三档098": "#c33", "+MA20+止损8": "#f66",
              "+MA20+止损12": "#fa8", "+MA20+对冲07": "#36c", "+MA20+对冲07B": "#39f",
              "+MA20+对冲10B": "#26e", "+MA20+止损8+对冲07B": "#a4c"}
    x_all = sorted(nav_series[labels[0]].index)
    for lb in labels:
        s = nav_series[lb].reindex(x_all).ffill().fillna(1.0)
        ax.plot(np.arange(len(x_all)), s.values, label=lb, lw=1.6, color=colors.get(lb, "#333"))
    ax.plot(np.arange(len(x_all)), (1 + bm_e_m.reindex(x_all).fillna(0)).cumprod().values,
            label="512100ETF", lw=1.2, ls="--", color="#999")
    ax.set_yscale("log")
    ax.set_ylabel("净值(对数)")
    ax.set_title("回撤控制补充: 个股止损 / IM空头对冲 (BASE+VAL+RS12+MA20, 2020-2026)", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(len(x_all)), 0, 1,
                     where=sig_rs12.reindex(x_all).fillna(False).values, color="#0c6", alpha=0.15)
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)
    fp = os.path.join(OUT_DIR, "risk_control_nav2.png")
    plt.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[保存图] {fp}")


if __name__ == "__main__":
    main()
