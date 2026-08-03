# -*- coding: utf-8 -*-
"""IM 股指期货贴水滚仓收益测算 → 校准中证1000增强的 α 假设 (2026-07-17)

方法:
  多头持有 IM 主力连续合约的总收益 ≈ 现货收益 + 贴水收敛收益(滚仓捕获)
  roll_capture(区间) = IM主力连续累计收益 - 中证1000现货累计收益
  按自然年分段年化, 得到贴水对"增强α"的可数据支撑地板值

数据: akshare futures_main_sina(IM0) + index_zh_a_hist(000852)
"""
import akshare as ak
import pandas as pd
import numpy as np

# ---------- 数据 ----------
fut = pd.read_csv(r"C:\Users\liuqi\quant_system_v2\im_main_sina.csv")
fut["date"] = pd.to_datetime(fut["日期"])
fut = fut.set_index("date")[["收盘价"]].rename(columns={"收盘价": "fut"})

# 现货: iFinD 000852.SH (两段拼接)
sa = pd.read_csv(r"C:\Users\liuqi\quant_system_v2\zz1000_spot_a.csv")
sb = pd.read_csv(r"C:\Users\liuqi\quant_system_v2\zz1000_spot_b.csv")
spot = pd.concat([sa, sb], ignore_index=True)
spot["date"] = pd.to_datetime(spot["time"])
spot = spot.drop_duplicates("date").set_index("date")[["close"]].rename(columns={"close": "spot"})

df = fut.join(spot, how="inner").dropna()
print(f"对齐样本: {len(df)} 天, {df.index.min().date()} ~ {df.index.max().date()}")

df["fut_ret"] = df["fut"].pct_change()
df["spot_ret"] = df["spot"].pct_change()
df["basis_pct"] = (df["fut"] / df["spot"] - 1) * 100  # 主力合约基差水平(参考)

# ---------- 分年滚仓捕获 ----------
print("\n分年滚仓捕获 (IM多头相对现货的超额):")
print(f"{'年份':6s} {'期货收益':>10s} {'现货收益':>10s} {'滚仓捕获':>10s} {'年化':>8s} {'交易日':>6s}")
rows = []
for yr, g in df.groupby(df.index.year):
    fr = (1 + g["fut_ret"].dropna()).prod() - 1
    sr = (1 + g["spot_ret"].dropna()).prod() - 1
    cap = fr - sr
    n = len(g)
    ann = (1 + cap) ** (242 / n) - 1 if n > 20 else np.nan
    rows.append((yr, fr, sr, cap, ann, n))
    print(f"{yr:<6d} {fr:>+9.2%} {sr:>+9.2%} {cap:>+9.2%} {ann:>+7.2%} {n:>6d}")

# 全区间
fr = (1 + df["fut_ret"].dropna()).prod() - 1
sr = (1 + df["spot_ret"].dropna()).prod() - 1
cap = fr - sr
n = len(df)
ann_full = (1 + cap) ** (242 / n) - 1
print(f"{'全区间':6s} {fr:>+9.2%} {sr:>+9.2%} {cap:>+9.2%} {ann_full:>+7.2%} {n:>6d}")

# 基差水平描述
print(f"\n主力合约基差水平: 均值 {df['basis_pct'].mean():+.2f}%, "
      f"中位数 {df['basis_pct'].median():+.2f}%, "
      f"区间 [{df['basis_pct'].min():+.2f}%, {df['basis_pct'].max():+.2f}%]")
print("近一年月度均基差(%):")
print(df["basis_pct"].resample("ME").mean().tail(12).round(2).to_string())

# 校准结论
roll_vals = [r[4] for r in rows if not np.isnan(r[4])]
print("\n" + "=" * 60)
print("校准结论:")
print(f"  贴水滚仓年化(各年): {', '.join(f'{v:+.1%}' for v in roll_vals)}")
print(f"  贴水滚仓年化(全区间): {ann_full:+.2%}")
print(f"  vs step13 硬编码假设 alpha_annual = 11%")
df.to_csv(r"C:\Users\liuqi\quant_system_v2\im_basis_analysis.csv")
print("已保存 im_basis_analysis.csv")
