# -*- coding: utf-8 -*-
"""P3 负面超跌反转 - 前视偏差检验 (2026-07-17 评审复核)

检验问题:
  报告中的 T+5/T+10 收益是从新闻日 T 起算, 还是合法地"观察到 T+1~T+3 不跌之后"
  (即 T+3 收盘买入) 起算?

方法:
  1) 复现报告分组统计 (重大负面: net_sentiment < -1 且 negative >= 2)
  2) 计算从 T+3 收盘后起算的持有收益:
       r_3to5  = (1+ret_5d/100)/(1+ret_3d/100)  - 1
       r_3to10 = (1+ret_10d/100)/(1+ret_3d/100) - 1
     若信号在 T+3 入场后仍然显著 -> 非前视; 若大部分收益来自 T~T+3 -> 有前视成分
"""
import pandas as pd
import numpy as np
from math import erf, sqrt

def welch_t(a, b):
    """Welch 双样本 t 检验, 返回 (t, p); 大样本用正态近似 p 值"""
    a, b = pd.Series(a).dropna().values, pd.Series(b).dropna().values
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    t = (ma - mb) / sqrt(va / na + vb / nb)
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, p

def one_samp_t(a, mu=0.0):
    """单样本 t 检验, 返回 (t, p); 大样本用正态近似 p 值"""
    a = pd.Series(a).dropna().values
    t = (a.mean() - mu) / (a.std(ddof=1) / sqrt(len(a)))
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, p

df = pd.read_csv(r"C:\Users\liuqi\quant_system_v2\negative_reversal_analysis.csv")
print(f"总样本: {len(df)}")

# 重大负面定义 (与报告一致)
major_neg = df[(df["net_sentiment"] < -1) & (df["negative"] >= 2)].copy()
print(f"重大负面样本: {len(major_neg)}")

# 分组 (drop_t1_t3 单位为 %)
g_signal = major_neg[major_neg["drop_t1_t3"] > -2]     # 负面后不跌
g_up     = major_neg[major_neg["drop_t1_t3"] > 0]      # 负面后反涨
g_crash  = major_neg[major_neg["drop_t1_t3"] < -5]     # 负面后大跌
g_rest   = major_neg[(major_neg["drop_t1_t3"] <= -2) & (major_neg["drop_t1_t3"] >= -5)]

def add_post_entry(g):
    g = g.copy()
    g["r_3to5"]  = ((1 + g["ret_5d"]  / 100) / (1 + g["ret_3d"] / 100) - 1) * 100
    g["r_3to10"] = ((1 + g["ret_10d"] / 100) / (1 + g["ret_3d"] / 100) - 1) * 100
    return g

g_signal, g_up, g_crash, g_rest = map(add_post_entry, [g_signal, g_up, g_crash, g_rest])

def show(name, g):
    print(f"\n[{name}] n={len(g)}")
    print(f"  报告口径(从T起算):  T+5 {g['ret_5d'].mean():+.2f}%  T+10 {g['ret_10d'].mean():+.2f}%  "
          f"T+5胜率 {(g['ret_5d']>0).mean()*100:.1f}%")
    print(f"  T+3收盘后入场口径:  T+3~T+5 {g['r_3to5'].mean():+.2f}%  T+3~T+10 {g['r_3to10'].mean():+.2f}%  "
          f"胜率 {(g['r_3to5']>0).mean()*100:.1f}%")
    return g

show("负面后不跌 (> -2%)", g_signal)
show("负面后反涨 (> 0%)",  g_up)
show("负面后大跌 (< -5%)", g_crash)
show("中间组 (-5% ~ -2%)", g_rest)

# 信号组 vs 大跌组: 两种口径的 t 检验
for col, label in [("ret_5d", "报告口径 T+5"), ("r_3to5", "T+3入场后 T+3~T+5")]:
    t, p = welch_t(g_signal[col], g_crash[col])
    print(f"\n信号组 vs 大跌组 [{label}]: t={t:.2f}, p={p:.2e}")

# 信号组 T+3入场后收益 vs 0 的单样本检验
t1, p1 = one_samp_t(g_signal["r_3to5"], 0)
t2, p2 = one_samp_t(g_signal["r_3to10"], 0)
print(f"\n信号组 T+3~T+5  均值 vs 0: t={t1:.2f}, p={p1:.4f}")
print(f"信号组 T+3~T+10 均值 vs 0: t={t2:.2f}, p={p2:.4f}")

# 前视成分分解: 收益中有多少来自 T~T+3 (选择窗口本身)
sel_win = g_signal["ret_3d"].mean()
tot5 = g_signal["ret_5d"].mean()
print(f"\n[前视成分分解] 信号组 T~T+3(选择窗口)均值 {sel_win:+.2f}%, "
      f"占 T+5 总收益 {tot5:+.2f}% 的 {sel_win/tot5*100 if tot5 else float('nan'):.0f}%")
