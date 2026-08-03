# -*- coding: utf-8 -*-
"""
筹码边际因子研究 - 因子构建 + IC 检验 + 回测（PIT 严格对齐）

因子: chip_momentum_q = -(股东户数环比变化率)
  - 股东户数减少(变化率为负) → 筹码集中 → factor 为正 → 预期未来收益高
PIT: 仅使用 ann_date(公告日) <= 调仓日 的报告期数据; 环比需两期均已公告
未来收益: 用本地 data_day1 的 pct_chg 累乘(自动含复权), 避免除权跳空

用法:
    python research/chip_momentum/run_study.py
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

DAY_DIR = settings.daily_data_path      # data_day1
_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HN_DIR = os.path.join(_DATA_ROOT, "holdernumber")
IW_DIR = os.path.join(_DATA_ROOT, "index_weight")
IDX_DIR = os.path.join(_DATA_ROOT, "index_daily")
START_YEAR = 2020
FORWARD_DAYS = 20          # 未来收益窗口(交易日)
COST_BPS = 20              # 双边成本 20bps (沿用已确认研究口径)
TOP_N = 50                 # 组合持股数
# 因子去极值: winsorize 分位数
WINSOR = (0.01, 0.99)


def load_trade_dates():
    """从 data_day1 文件名获取全部交易日(升序)"""
    dates = sorted(f[:8] for f in os.listdir(DAY_DIR) if f.endswith(".parquet"))
    return dates


def load_holder_panel():
    """读全部股东户数, 返回 DataFrame[ts_code, ann_date, end_date, holder_num]"""
    frames = []
    for f in os.listdir(HN_DIR):
        if not f.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(HN_DIR, f))
        if df.empty:
            continue
        frames.append(df[["ts_code", "ann_date", "end_date", "holder_num"]])
    if not frames:
        raise RuntimeError("holdernumber 数据为空, 请先运行 fetch_data.py")
    panel = pd.concat(frames, ignore_index=True)
    panel["ann_date"] = panel["ann_date"].astype(str)
    panel["end_date"] = panel["end_date"].astype(str)
    panel["holder_num"] = pd.to_numeric(panel["holder_num"], errors="coerce")
    panel = panel.dropna(subset=["holder_num"])
    panel = panel.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    return panel


def load_index_weight_month(date_str):
    """取 <= date 的最近一个月成分股, 返回 set"""
    iw_dates = sorted(f[3:11] for f in os.listdir(IW_DIR) if f.startswith("iw_"))
    avail = [d for d in iw_dates if d <= date_str]
    if not avail:
        return set()
    df = pd.read_parquet(os.path.join(IW_DIR, f"iw_{avail[-1]}.parquet"))
    return set(df["con_code"].astype(str).str.strip())


def winsorize(s):
    lo, hi = s.quantile(WINSOR[0]), s.quantile(WINSOR[1])
    return s.clip(lo, hi)


def zscore(s):
    return (s - s.mean()) / s.std()


def newey_west_t(ics, lag=4):
    """Newey-West 修正 t 值 (样本自相关)"""
    ics = np.asarray(ics, dtype=float)
    n = len(ics)
    if n < 2:
        return 0.0, 0.0
    mean = ics.mean()
    var = ics.var(ddof=1)
    for l in range(1, min(lag, n - 1) + 1):
        cov = np.cov(ics[:-l], ics[l:], ddof=1)[0, 1]
        var += 2 * (1 - l / (lag + 1)) * cov
    se = np.sqrt(max(var, 1e-12) / n)
    return mean / se, mean / var if var > 0 else 0.0


def build_monthly_panel(holder, trade_dates):
    """构建月度因子面板: 每个月末调仓日的横截面
    返回: dict date -> DataFrame(index=ts_code, factor, fwd_ret)
    """
    # 调仓日: 每月最后一个交易日, 从 START_YEAR 起
    months = {}
    for d in trade_dates:
        if d[:4] >= str(START_YEAR):
            months[d[:6]] = d
    rebal_dates = sorted(months.values())
    # 移除最后一个调仓日(未来收益不足)
    rebal_dates = rebal_dates[:-1]

    # 预读所有窗口日 pct_chg (pivot: date x ts_code)
    start_dt = trade_dates.index(rebal_dates[0])
    end_dt = trade_dates.index(rebal_dates[-1]) + FORWARD_DAYS + 1
    win_dates = trade_dates[start_dt:end_dt]
    print(f"[load] 交易日窗口: {win_dates[0]} ~ {win_dates[-1]} ({len(win_dates)} days)")
    ret_panels = []
    for i, d in enumerate(win_dates):
        fp = os.path.join(DAY_DIR, f"{d}.parquet")
        df = pd.read_parquet(fp, columns=["ts_code", "pct_chg"])
        df = df.set_index("ts_code")
        df.columns = [d]
        ret_panels.append(df)
    ret_panel = pd.concat(ret_panels, axis=1)
    ret_panel = ret_panel.astype(float) / 100.0
    print(f"[load] pct_chg panel: {ret_panel.shape}")

    holder = holder.set_index("ts_code")
    out = {}
    for T in rebal_dates:
        # 1) 股票池
        universe = load_index_weight_month(T)
        # 2) 每只股票: 截至 T 已公告的最新两期
        rows = []
        for code in universe:
            if code not in holder.index:
                continue
            h = holder.loc[code]
            if isinstance(h, pd.DataFrame):
                h = h[h["ann_date"] <= T]
                if h.empty:
                    continue
                h = h.iloc[-1]
            else:
                if h["ann_date"] > T:
                    continue
            rows.append((code, h["ann_date"], h["end_date"], h["holder_num"]))
        if not rows:
            continue
        cur = pd.DataFrame(rows, columns=["ts_code", "ann_date", "end_date", "holder_num"])
        # 3) 环比: 需要上一报告期(ann_date<=T 且 end_date 更早) — 用 per-code 滚动
        # 简化: 对每只股票取已公告序列倒数两项
        cur_list = []
        for code, grp in holder[holder.index.isin(cur["ts_code"])].groupby(level=0):
            g = grp[grp["ann_date"] <= T]
            if len(g) < 2:
                continue
            g = g.iloc[-2:]
            cur_list.append({
                "ts_code": code,
                "h_t": g["holder_num"].iloc[-1],
                "h_t1": g["holder_num"].iloc[0],
            })
        if not cur_list:
            continue
        fdf = pd.DataFrame(cur_list).set_index("ts_code")
        fdf["chip_chg"] = (fdf["h_t"] - fdf["h_t1"]) / fdf["h_t1"]
        fdf["factor"] = -fdf["chip_chg"]   # 筹码集中为正
        # 4) 未来收益: T 之后第1至第1+FORWARD_DAYS 个交易日
        idx = trade_dates.index(T)
        fwd_dates = trade_dates[idx + 1: idx + 1 + FORWARD_DAYS]
        r = ret_panel.loc[fdf.index, fwd_dates]
        fdf["fwd_ret"] = (1 + r).prod(axis=1) - 1
        fdf = fdf.dropna(subset=["factor", "fwd_ret"])
        fdf = fdf[fdf["fwd_ret"] > -0.95]
        if len(fdf) > 20:
            out[T] = fdf[["factor", "fwd_ret"]]
    print(f"[panel] {len(out)} monthly cross-sections built")
    return out, ret_panel


def run_ic(panel):
    print("\n" + "=" * 70)
    print("Rank IC 检验 (chip_momentum_q, 未来20交易日收益)")
    print("=" * 70)
    rows = []
    for T, df in sorted(panel.items()):
        ic = df["factor"].rank().corr(df["fwd_ret"].rank())
        rows.append({"date": T, "ic": ic, "n": len(df)})
    ic_df = pd.DataFrame(rows)
    ics = ic_df["ic"].dropna().values
    t_nw, _ = newey_west_t(ics)
    print(f"截面数: {len(ics)}  | 每期平均样本: {ic_df['n'].mean():.0f}")
    print(f"IC 均值: {ics.mean():.4f}  | IC 标准差: {ics.std(ddof=1):.4f}  | ICIR: {ics.mean()/ics.std(ddof=1):.4f}")
    print(f"NW t值(lag=4): {t_nw:.3f}  | 正IC占比: {(ics>0).mean():.2%}")
    print("近12期IC:", [f"{v:.3f}" for v in ics[-12:]])
    # 分组收益
    print("\n分5组检验 (Q1=筹码最分散, Q5=筹码最集中):")
    grp_rows = []
    for T, df in sorted(panel.items()):
        df = df.copy()
        df["grp"] = pd.qcut(df["factor"], 5, labels=False, duplicates="drop") + 1
        g = df.groupby("grp")["fwd_ret"].mean()
        grp_rows.append(g)
    grp_df = pd.DataFrame(grp_rows)
    print(grp_df.describe().loc[["mean", "std"]].to_string())
    if len(grp_df.columns) >= 5:
        spread = grp_df[grp_df.columns.max()] - grp_df[grp_df.columns.min()]
        print(f"\nQ5-Q1 月均价差: {spread.mean()*100:.2f}pp  | 价差ICIR: {spread.mean()/spread.std(ddof=1):.3f}")
    ic_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ic_monthly.csv"), index=False)
    return ic_df


def run_backtest(panel, ret_panel, trade_dates):
    print("\n" + "=" * 70)
    print(f"回测: 月度调仓 Top{TOP_N} 等权 (成本 {COST_BPS}bps 双边), 基准=中证1000")
    print("=" * 70)
    # 组合净值
    nav = 1.0
    navs, dates_l = [], []
    # 基准: 中证1000 指数日收益
    idx_path = os.path.join(IDX_DIR, "000852.SH.parquet")
    idx = pd.read_parquet(idx_path).set_index("trade_date")["pct_chg"].astype(float) / 100.0
    idx_nav = 1.0
    pos = []
    last_T = None
    for T, df in sorted(panel.items()):
        df = df.sort_values("factor", ascending=False)
        picks = df.head(TOP_N).index.tolist()
        idx_d = trade_dates.index(T)
        hold_dates = trade_dates[idx_d + 1: idx_d + 1 + FORWARD_DAYS]
        # 月度组合收益: 日收益累乘后减成本
        r = ret_panel.loc[picks, hold_dates]
        port_ret = r.mean(axis=0)
        cost = COST_BPS / 10000.0
        gross = float((1 + port_ret).prod()) - 1.0
        net = gross - cost
        nav *= (1 + net)
        # 基准收益(同窗口)
        idx_window = [d for d in hold_dates if d in idx.index]
        if idx_window:
            idx_nav *= float((1 + idx.loc[idx_window]).prod())
        navs.append(nav)
        dates_l.append(T)
        pos.append(len(picks))
        last_T = T

    nav_s = pd.Series(navs, index=dates_l)
    # 指标
    mrets = nav_s.pct_change().dropna()
    n = len(mrets)
    ann_ret = nav_s.iloc[-1] ** (12 / n) - 1
    ann_vol = mrets.std(ddof=1) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    dd = (nav_s / nav_s.cummax() - 1).min()
    idx_nav_final = idx_nav
    idx_ann = idx_nav_final ** (12 / n) - 1
    print(f"区间: {dates_l[0]} ~ {dates_l[-1]}  (共 {n} 个月)")
    print(f"组合: 累计 {nav_s.iloc[-1]:.3f}  | 年化 {ann_ret*100:.2f}%  | Sharpe {sharpe:.2f}  | MaxDD {dd*100:.1f}%")
    print(f"基准: 累计 {idx_nav_final:.3f}  | 年化 {idx_ann*100:.2f}%")
    print(f"超额年化: {(ann_ret - idx_ann)*100:.2f}pp  | 月胜率: {(mrets > 0).mean():.2%}")
    nav_s.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nav.csv"))
    return nav_s


def ensure_index_data():
    """拉取中证1000指数日行情(基准)"""
    os.makedirs(IDX_DIR, exist_ok=True)
    fp = os.path.join(IDX_DIR, "000852.SH.parquet")
    if os.path.exists(fp):
        return
    import tushare as ts
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    df = pro.index_daily(ts_code="000852.SH", start_date=f"{START_YEAR}0101",
                         end_date="20260731")
    df.to_parquet(fp)
    print(f"[index] saved {len(df)} rows")


def main():
    t0 = datetime.now()
    ensure_index_data()
    trade_dates = load_trade_dates()
    print(f"[dates] {len(trade_dates)} trading days")
    holder = load_holder_panel()
    print(f"[holder] {len(holder)} rows, {holder['ts_code'].nunique()} stocks, "
          f"end_date range {holder['end_date'].min()}~{holder['end_date'].max()}")
    panel, ret_panel = build_monthly_panel(holder, trade_dates)
    ic_df = run_ic(panel)
    nav_s = run_backtest(panel, ret_panel, trade_dates)
    print(f"\n[done] elapsed {datetime.now()-t0}")


if __name__ == "__main__":
    main()
