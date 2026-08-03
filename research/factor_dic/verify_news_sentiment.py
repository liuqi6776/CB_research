# -*- coding: utf-8 -*-
"""news_sentiment 验证 v2: 用正确的 ret_1d/ret_2d/... 收益列"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_csv(os.path.join(BASE, "news_academic_full.csv"))
print("rows:", len(df), " unique dates:", df["trade_date"].nunique(),
      " range:", df["trade_date"].min(), "->", df["trade_date"].max())


def nw_t(x, y, lag):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = len(x)
    xm = x - x.mean()
    ym = y - y.mean()
    beta = np.dot(xm, ym) / np.dot(xm, xm)
    e = ym - beta * xm
    s = xm * e
    g0 = np.dot(s, s) / m
    se2 = g0
    for j in range(1, min(lag + 1, m - 1)):
        gj = np.dot(s[:-j], s[j:]) / m
        se2 += 2 * (1 - j / (lag + 1)) * gj
    t = beta / np.sqrt(se2 / m)
    r = np.corrcoef(x, y)[0, 1]
    return r, t


out = []
out.append("rows: %d, unique dates: %d, range: %s -> %s" % (
    len(df), df["trade_date"].nunique(), df["trade_date"].min(), df["trade_date"].max()))
out.append("")
out.append("=== 逐条新闻 net_sentiment vs 前瞻收益 (Newey-West lag=持有期) ===")
out.append("%-8s %6s %10s %8s %s" % ("窗口", "n", "r", "NWt", "显著?"))
print("%-8s %6s %10s %8s %s" % ("窗口", "n", "r", "NWt", "显著?"))
for col, lag in [("ret_1d", 1), ("ret_2d", 2), ("ret_3d", 3), ("ret_5d", 5), ("ret_10d", 10)]:
    sub = df[["net_sentiment", col]].dropna()
    r, t = nw_t(sub["net_sentiment"], sub[col], lag)
    sig = "显著" if abs(t) > 1.96 else "不显著"
    line = "%-8s %6d %10.4f %8.2f %s" % (col, len(sub), r, t, sig)
    out.append(line)
    print(line)

out.append("")
out.append("=== 负面情感 negative vs 前瞻收益 ===")
out.append("%-8s %6s %10s %8s %s" % ("窗口", "n", "r", "NWt", "显著?"))
print("%-8s %6s %10s %8s %s" % ("窗口", "n", "r", "NWt", "显著?"))
for col, lag in [("ret_1d", 1), ("ret_2d", 2), ("ret_3d", 3)]:
    sub = df[["negative", col]].dropna()
    r, t = nw_t(sub["negative"], sub[col], lag)
    sig = "显著" if abs(t) > 1.96 else "不显著"
    line = "%-8s %6d %10.4f %8.2f %s" % (col, len(sub), r, t, sig)
    out.append(line)
    print(line)

# 每日聚合
g = df.groupby("trade_date").agg(
    net=("net_sentiment", "mean"), neg=("negative", "mean"),
    r1=("ret_1d", "mean"), r2=("ret_2d", "mean"), r3=("ret_3d", "mean")).dropna()
out.append("")
out.append("=== 每日聚合层面 (n=%d 天) ===" % len(g))
out.append("%-12s %10s %8s %s" % ("变量", "r", "t", "显著?"))
print("%-12s %10s %8s %s" % ("变量", "r", "t", "显著?"))
for a, b in [("net", "r1"), ("net", "r2"), ("net", "r3"), ("neg", "r1")]:
    r = np.corrcoef(g[a], g[b])[0, 1]
    n = len(g)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    sig = "显著" if abs(t) > 1.96 else "不显著"
    line = "%-12s %10.4f %8.2f %s" % (a + " vs " + b, r, t, sig)
    out.append(line)
    print(line)

# 多空策略
ls = pd.read_csv(os.path.join(BASE, "daily_ls_strategy.csv"))
ls3 = ls[ls["window"] == 3]
r_ = ls3["ls_ret"].values
n = len(r_)
mean = r_.mean()
sh = mean / r_.std(ddof=1)
t_nw = mean / (r_.std(ddof=1) / np.sqrt(n)) * np.sqrt(n) / np.sqrt(n)  # placeholder
e = r_ - mean
m = n
g0 = np.dot(e, e) / m
se2 = g0
for j in range(1, 3):
    gj = np.dot(e[:-j], e[j:]) / m
    se2 += 2 * (1 - j / 3) * gj
t_nw = mean / np.sqrt(se2 / m)
out.append("")
out.append("=== 多空策略 T+3 (n=%d) ===" % n)
out.append("mean=%s 日频Sharpe=%s" % (format(mean, ".3%"), format(sh, ".3f")))
out.append("朴素t=%.2f  NeweyWest t(lag=2)=%.2f  %s" % (sh * np.sqrt(n), t_nw,
    "显著" if abs(t_nw) > 1.96 else "不显著"))
print("多空: mean=%s Sharpe=%s 朴素t=%.2f NWt=%.2f" % (format(mean, ".3%"), format(sh, ".3f"), sh * np.sqrt(n), t_nw))

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "news_sentiment_verify.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("[保存] results/news_sentiment_verify.txt")
