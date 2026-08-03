# -*- coding: utf-8 -*-
"""
方向验证: RS12 弱市段"持 512100"改为持避险资产 (货基/国债/黄金/弱段动量), 是否更优

主策略: BASE+VAL Top50 + RS12 + MA20三档(0.98), 2020-2026
弱段(rs12_off)持有资产变体:
  V0 现状:     512100 (中证1000 ETF, 弱段也跌 -> 当前回撤下限来源)
  V1 货币ETF:  511990 价格 + 1.8%/年分红假设 (零回撤)
  V2 国债ETF:  511260 十年国债 (MaxDD 4.6%)
  V3 黄金ETF:  518880 (波动大, 检验是否真避险)
  V4 弱段动量: 弱段调仓日选过去20日最强避险资产 (货基/国债/黄金) 持有

数据: research/serve/data/etf/{code}.parquet (工作区实际使用 tushare fund_daily 拉取;
      fetch_defensive_etf.py 为 akshare 备选脚本, 两者数据源/路径需对齐)
输出: research/factor_dic/results/defensive_asset_bt.txt
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic import style_factors as sf
from research.factor_dic import risk_control_bt as rcb

ETF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serve", "data", "etf")
OUT_DIR = rv.OUT_DIR
DEEP = 0.98
MM_ANNUAL = 0.018  # 货币ETF 分红年化假设
COST = rv.COST_BPS / 10000.0  # 20bps 双边


def load_asset_ret(code, idx_dir=False):
    if idx_dir:
        # 512100 在 IDX_DIR, 已有 pct_chg 列
        df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
        df["trade_date"] = df["trade_date"].astype(str).str[:8]
        df = df.set_index("trade_date").sort_index()
        return df["pct_chg"] / 100.0
    fp = os.path.join(ETF_DIR, f"{code}.parquet")
    df = pd.read_parquet(fp)
    df = df.reset_index()
    if "trade_date" not in df.columns:
        df.columns = ["trade_date", "close"]
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    s = df.set_index("trade_date")["close"].pct_change().dropna()
    return s


def stats(pr, per_year=242.0):
    pr = pr.dropna()
    if len(pr) < 30:
        return None
    nav = (1 + pr).cumprod()
    years = len(pr) / per_year
    cagr = nav.iloc[-1] ** (1 / years) - 1
    sharpe = pr.mean() / (pr.std(ddof=1) + 1e-12) * np.sqrt(per_year)
    mdd = ((nav.cummax() - nav) / nav.cummax()).max()
    return cagr, sharpe, mdd, cagr / mdd if mdd > 0 else np.nan


def monthly_rebalanced(weights, grid, rebal, dates_idx):
    """月度再平衡混合: 每个调仓日重置为目标权重, 持有期内按资产收益自然漂移
    (修复: 不再按每日恒权相加, 后者隐含每日再平衡且未计再平衡成本)"""
    keys = list(weights)
    rp = np.zeros(len(dates_idx))
    for i, rb in enumerate(rebal):
        rb_next = rebal[i + 1] if i + 1 < len(rebal) else dates_idx[-1]
        lo = dates_idx.searchsorted(rb, side="right")
        hi = dates_idx.searchsorted(rb_next, side="right")
        w = dict(weights)  # 调仓日重置目标权重
        for j in range(lo, hi):
            r_t = sum(w[k] * grid[k][j] for k in keys)
            rp[j] = r_t
            for k in keys:
                w[k] = w[k] * (1.0 + grid[k][j]) / (1.0 + r_t)
    return rp


def main():
    trade_dates = rv.load_trade_dates()
    dates_idx = pd.Index(trade_dates)
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

    sml = rcb.load_idx("000852.SH")
    big = rcb.load_idx("000300.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)
    idx_close = sml["close"]
    ma20 = idx_close.rolling(20).mean()
    # MA20 三档仓位: 用 T-1 日收盘信号, T 日生效 (修复同日信号前视)
    ma20_w = pd.Series(1.0, index=sml.index)
    c, m = idx_close.shift(1), ma20.shift(1)
    ma20_w[c < m] = 0.5
    ma20_w[c < DEEP * m] = 0.0

    # 弱段资产日收益 (对齐到 trade_dates)
    etf1000 = load_asset_ret("512100.SH", idx_dir=True)
    mm = load_asset_ret("511990.SH") + MM_ANNUAL / 242.0
    bond = load_asset_ret("511260.SH")
    gold = load_asset_ret("518880.SH")

    def asset_grid(dates):
        # 缺失日填 0 (NAV 不变), 不做 ffill 复制上一日收益 (修复 ffill 语义错误)
        return {d: a.reindex(dates).fillna(0.0).values for d, a in [
            ("512100", etf1000), ("货基", mm), ("国债", bond), ("黄金", gold)]}

    grid = asset_grid(dates_idx)
    ma20_w_arr = ma20_w.reindex(dates_idx).ffill().fillna(1.0)

    # 弱段动量: 调仓日选过去20日最强(货基/国债/黄金), 持有到下次调仓
    mom_series = pd.Series(0.0, index=dates_idx)
    for i, rb in enumerate(rebal):
        if sig_rs12.loc[rb]:
            continue
        rb_next = rebal[i + 1] if i + 1 < len(rebal) else dates_idx[-1]
        prev = dates_idx[dates_idx < rb][-20:]
        if len(prev) < 5:
            continue
        scores = {}
        for nm, a in [("货基", mm), ("国债", bond), ("黄金", gold)]:
            r = a.reindex(prev)
            # 动量: 过去20日复利累计收益 (1+r).prod()-1 (修复直接 prod 的错误)
            scores[nm] = float((1.0 + r.fillna(0.0)).prod() - 1.0)
        best = max(scores, key=scores.get)
        seg = dates_idx[(dates_idx > rb) & (dates_idx <= rb_next)]
        if len(seg):
            src = {"货基": mm, "国债": bond, "黄金": gold}[best]
            mom_series.loc[seg] = src.reindex(seg).fillna(0.0).values

    # 混合变体: 月度再平衡 (期初等权、期内按收益漂移、调仓日重置)
    mix5 = monthly_rebalanced({"货基": 0.5, "国债": 0.5}, grid, rebal, dates_idx)
    mix6 = monthly_rebalanced({"货基": 0.25, "国债": 0.75}, grid, rebal, dates_idx)
    mix7 = monthly_rebalanced({"货基": 0.75, "国债": 0.25}, grid, rebal, dates_idx)
    mix8 = monthly_rebalanced({"货基": 1 / 3, "国债": 1 / 3, "黄金": 1 / 3}, grid, rebal, dates_idx)

    variants = [
        ("V0 现状: 弱段持512100", grid["512100"]),
        ("V1 货基ETF(零回撤)", grid["货基"]),
        ("V2 国债ETF(十年)", grid["国债"]),
        ("V3 黄金ETF", grid["黄金"]),
        ("V4 弱段动量(货基/国债/黄金)", mom_series.values),
        ("V5 货基50%+国债50%(月平衡)", mix5),
        ("V6 货基25%+国债75%(月平衡)", mix6),
        ("V7 货基75%+国债25%(月平衡)", mix7),
        ("V8 三资产等权(月平衡)", mix8),
        ("V8d 三资产等权(每日恒权,对照)", (grid["货基"] + grid["国债"] + grid["黄金"]) / 3.0),
    ]

    lines = []
    lines.append("=" * 96)
    lines.append("方向验证: RS12 弱段持避险资产 (BASE+VAL+RS12+MA20三档0.98, 2020-2026)")
    lines.append(f"弱段口径: rs12_off 时持目标资产全额; 强段 = 组合 × MA20三档(deep={DEEP})")
    lines.append(f"货基收益: 511990 价格 + {MM_ANNUAL:.1%}/年分红假设 | 数据源: tushare fund_daily")
    lines.append("混合变体口径: 月度再平衡 (调仓日重置目标权重, 期内按收益漂移); V8d 为每日恒权对照(隐含每日再平衡)")
    lines.append("=" * 96)

    navs = {}

    # 主循环: 逐调仓日生成组合日收益 (与 risk_control_bt.py 同口径)
    comb_map = {}
    for i, rb in enumerate(rebal):
        members = rv.load_index_weight(rb)
        fvals = {}
        for code in members or []:
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
            for pname in panels:
                p = panels[pname].get(rb)
                if p is not None and code in p.index:
                    v = p.loc[code]
                    if np.isfinite(v):
                        row[pname] = v
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < rv.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = sf.BASE_COLS + ["VAL"]
        has = zdf[cols].dropna()
        if len(has) < rv.TOP_N:
            continue
        scored = has.mean(axis=1).sort_values(ascending=False)
        top = scored.index[:rv.TOP_N].tolist()
        rb_next = rebal[i + 1] if i + 1 < len(rebal) else dates_idx[-1]
        hi, hn = dates_idx.searchsorted(rb), dates_idx.searchsorted(rb_next, side="right")
        hold = dates_idx[hi + 1:hn]
        if not len(hold):
            continue
        comb = pct_df.reindex(columns=top).reindex(hold).fillna(0.0) / 100.0
        comb_map[rb] = comb.mean(axis=1)

    for vname, varr in variants:
        pr = pd.Series(0.0, index=dates_idx)
        for i, rb in enumerate(rebal):
            rb_next = rebal[i + 1] if i + 1 < len(rebal) else dates_idx[-1]
            hi, hn = dates_idx.searchsorted(rb), dates_idx.searchsorted(rb_next, side="right")
            hold = dates_idx[hi + 1:hn]
            if not len(hold):
                continue
            if sig_rs12.loc[rb]:  # 强段: 组合 × MA20三档
                w_ = ma20_w_arr.reindex(hold).values
                pr.loc[hold] = (comb_map.get(rb, pd.Series(0.0, index=hold)) * w_).values
            else:  # 弱段: 持目标资产
                pr.loc[hold] = varr[dates_idx.get_indexer(hold)]
            pr.loc[hold[0]] = pr.loc[hold[0]] - COST  # 每次调仓 20bps

        pr = pr[pr.index >= rebal[0]]  # 去掉 2020 前的空段, 与主回测同起点
        navs[vname] = (1 + pr).cumprod()
        s = stats(pr)
        if s is None:
            continue
        cagr, sh, mdd, cm = s
        lines.append(f"{vname:<38} 年化{cagr:>7.2%}  Sharpe{sh:>6.2f}  MaxDD{mdd:>7.2%}  卡玛{cm:>6.2f}")

    # 分年度 (V0 vs V5 混合50/50)
    lines.append("\n分年度对比 (V0 现状 vs V5 货基50%+国债50%):")
    pr0 = pd.Series(0.0, index=dates_idx)
    pr1 = pd.Series(0.0, index=dates_idx)
    for i, rb in enumerate(rebal):
        rb_next = rebal[i + 1] if i + 1 < len(rebal) else dates_idx[-1]
        hi, hn = dates_idx.searchsorted(rb), dates_idx.searchsorted(rb_next, side="right")
        hold = dates_idx[hi + 1:hn]
        if not len(hold):
            continue
        if sig_rs12.loc[rb]:
            w_ = ma20_w_arr.reindex(hold).values
            pr0.loc[hold] = (comb_map.get(rb, pd.Series(0.0, index=hold)) * w_).values
            pr1.loc[hold] = pr0.loc[hold].values
        else:
            pr0.loc[hold] = grid["512100"][dates_idx.get_indexer(hold)]
            pr1.loc[hold] = mix5[dates_idx.get_indexer(hold)]
        pr0.loc[hold[0]] = pr0.loc[hold[0]] - COST
        pr1.loc[hold[0]] = pr1.loc[hold[0]] - COST
    pr0 = pr0[pr0.index >= rebal[0]]
    pr1 = pr1[pr1.index >= rebal[0]]
    for y in range(2020, 2027):
        ypr0 = pr0[pr0.index.str[:4] == str(y)]
        ypr1 = pr1[pr1.index.str[:4] == str(y)]
        if len(ypr0):
            r0, r1 = (1 + ypr0).prod() - 1, (1 + ypr1).prod() - 1
            lines.append(f"  {y}: 现状 {r0:>7.2%} | 混合50/50 {r1:>7.2%}")

    text = "\n".join(lines) + "\n"
    print(text)
    fp = os.path.join(OUT_DIR, "defensive_asset_bt.txt")
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[保存] {fp}")

    # ---------- 净值曲线图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(11, 5.5))
        # 10 变体 10 色 (修复: 原只有 5 色, V5-V8/V8d 画不出)
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for (nm, nav), cc in zip(navs.items(), colors):
            ax.plot(nav.index, nav.values, label=nm, color=cc, lw=1.4)
        ax.set_title("RS12 弱段持不同避险资产 - 净值曲线 (BASE+VAL+RS12+MA20三档0.98)")
        ax.set_ylabel("净值")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        figfp = os.path.join(OUT_DIR, "defensive_asset_bt.png")
        fig.savefig(figfp, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[保存] {figfp}")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")


if __name__ == "__main__":
    main()
