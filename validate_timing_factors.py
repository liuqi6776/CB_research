# -*- coding: utf-8 -*-
"""中证1000/沪深300 择时因子时间序列检查 (2026-07-17)

主线多资产+择时的核心信号:
  1. val_q = (PE 5年分位 + PB 5年分位)/2, 低估加仓 (1 - val_q)
  2. 趋势闸: close >= MA250 (跌破减半仓)
  3. 强制入场: val_q <= 15% 时无视趋势抄底 (原研究提示 A股存在价值陷阱)

检查(时间序列版五道关卡):
  - val_q vs 未来 20/60/120 日指数收益的 Spearman 相关 (期望: 负相关)
  - 分半稳定性: 2015-2020 vs 2021-2026
  - 趋势闸条件收益: MA上方 vs 下方 的未来收益差异
  - 强制入场规则: val_q<=15% 且 MA下方(价值陷阱区) vs val_q<=15% 且 MA上方
"""
import pandas as pd
import numpy as np
from math import sqrt
import sys
sys.path.insert(0, r"C:\Users\liuqi\quant_system_v2")
from utils.factor_checks import pval_norm, sig_stars, welch_t

DATA = r"C:\Users\liuqi\quant_system_v2\etf-valuation-strategy\data"

def spearman(a, b):
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 10:
        return np.nan, np.nan, len(x)
    r = x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank())
    # 时间序列自相关, 用 Newey-West 风格的保守 t (粗略: 按非重叠块折减有效样本)
    n = len(x)
    return r, n

def ts_corr_by_window(df, val_col, fwd_col, label):
    x = df[[val_col, fwd_col]].dropna()
    if len(x) < 30:
        print(f"    {label}: 样本不足({len(x)})")
        return
    # 非重叠块 (块长=fwd期限) 降低自相关
    blk = x.iloc[::BLK]
    r_full = x[val_col].rank().corr(x[fwd_col].rank())
    r_blk = blk[val_col].rank().corr(blk[fwd_col].rank())
    n_blk = len(blk)
    t = r_blk * sqrt((n_blk - 2) / max(1e-12, 1 - r_blk ** 2))
    p = pval_norm(t)
    print(f"    {label}: 全样本ρ={r_full:+.3f} | 非重叠块ρ={r_blk:+.3f}, t={t:+.2f}, "
          f"p={p:.3f} {sig_stars(p)}, n_blk={n_blk}")

BLK = 60  # 非重叠块长(与fwd60对应)

for name, pf, vf in [("中证1000", "zz1000_daily.csv", "zz1000_valuation.csv"),
                     ("沪深300", "hs300_daily.csv", "hs300_valuation.csv")]:
    print("=" * 64)
    print(f"[{name}]")
    px = pd.read_csv(f"{DATA}/{pf}")
    va = pd.read_csv(f"{DATA}/{vf}")
    px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str))
    va["trade_date"] = pd.to_datetime(va["trade_date"].astype(str))
    px = px.sort_values("trade_date").reset_index(drop=True)
    va = va.sort_values("trade_date").reset_index(drop=True)

    va["pe_q"] = va["pe_ttm"].rolling(1200, min_periods=250).rank(pct=True)
    va["pb_q"] = va["pb"].rolling(1200, min_periods=250).rank(pct=True)
    va["val_q"] = (va["pe_q"] + va["pb_q"]) / 2

    df = px.merge(va[["trade_date", "val_q"]], on="trade_date", how="inner")
    df["ma250"] = df["close"].rolling(250).mean()
    for h in [20, 60, 120]:
        df[f"fwd{h}"] = df["close"].shift(-h) / df["close"] - 1
    df = df.dropna(subset=["val_q", "ma250", "fwd60"]).reset_index(drop=True)
    print(f"  样本: {len(df)} 天, {df['trade_date'].min().date()}~{df['trade_date'].max().date()}")

    # 1. val_q 预测力 (期望负相关: 估值越低未来收益越高)
    print("  val_q vs 未来收益 (负相关=有效):")
    for h in [20, 60, 120]:
        ts_corr_by_window(df, "val_q", f"fwd{h}", f"fwd{h}")

    # 分半
    med = df["trade_date"].median()
    print("  分半 (fwd60):")
    for label, wd in [("前半", df[df["trade_date"] <= med]), ("后半", df[df["trade_date"] > med])]:
        ts_corr_by_window(wd, "val_q", "fwd60", label)

    # 2. 趋势闸
    up = df[df["close"] >= df["ma250"]]["fwd60"] * 100
    dn = df[df["close"] < df["ma250"]]["fwd60"] * 100
    t, p = welch_t(up, dn)
    print(f"  趋势闸: MA上方 fwd60 {up.mean():+.2f}% (n={len(up)}) vs MA下方 {dn.mean():+.2f}% (n={len(dn)}), "
          f"t={t:+.2f}, p={p:.3f} {sig_stars(p)}")

    # 3. 强制入场规则 (价值陷阱检验): MA下方时, 极低估值 vs 非极低估值
    below = df[df["close"] < df["ma250"]]
    cheap = below[below["val_q"] <= 0.15]["fwd60"] * 100
    notcheap = below[below["val_q"] > 0.15]["fwd60"] * 100
    t2, p2 = welch_t(cheap, notcheap)
    print(f"  强制入场(MA下方): val_q<=15% fwd60 {cheap.mean():+.2f}% (n={len(cheap)}) vs "
          f">15% {notcheap.mean():+.2f}% (n={len(notcheap)}), t={t2:+.2f}, p={p2:.3f} {sig_stars(p2)}")

    # 4. val_q 五档条件收益
    df["vq_bin"] = pd.qcut(df["val_q"], 5, labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
    piv = df.groupby("vq_bin", observed=True)["fwd60"].mean() * 100
    cnt = df.groupby("vq_bin", observed=True)["fwd60"].count()
    print("  val_q五档 fwd60(%): " + ", ".join(f"{k}={v:+.2f}(n={cnt[k]})" for k, v in piv.items()))
