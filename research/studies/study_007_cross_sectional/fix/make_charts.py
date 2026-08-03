# -*- coding: utf-8 -*-
"""
make_charts.py — study_007 修复版图表生成（托管 Python 运行）
=============================================================
用户 anaconda Python 的 matplotlib 依赖 PIL 而 PIL DLL 损坏（已实测），
故绘图在托管 Python（matplotlib 3.10.9 可用）下执行，输入为
results_fixed/ 下由 run_fixed.py（用户 Python）生成的 CSV。

运行: python make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

FIX = Path(__file__).resolve().parent
RES = FIX / "results_fixed"

C_STRAT = "#c0392b"   # 策略: 深红
C_BENCH = "#7f8c8d"   # 基准: 灰
C_Q1 = "#c0392b"
C_Q5 = "#2c3e50"


def load(name):
    return pd.read_csv(RES / name, dtype={"trade_date": str})


def to_dt(s):
    return pd.to_datetime(s, format="%Y%m%d")


# ---------------------------------------------------------------------------
# 图1: 主回测净值 vs 等权基准（对数轴，标注最大回撤区间）
# ---------------------------------------------------------------------------
df = load("main_nav_vs_benchmark.csv")
df["dt"] = to_dt(df["trade_date"])
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(df["dt"], df["nav"], color=C_STRAT, lw=1.6,
        label="修复后策略净值 (top50, 双边成本0.3%)")
ax.plot(df["dt"], df["bench_nav"], color=C_BENCH, lw=1.4,
        label="全A等权基准 (mkt_ret 复利)")
ax.set_yscale("log")
dd = df["nav"] / df["nav"].cummax() - 1.0
trough_i = int(dd.idxmin())
peak_i = int(df["nav"].iloc[:trough_i + 1].idxmax())
ax.axvspan(df["dt"].iloc[peak_i], df["dt"].iloc[trough_i],
           color="#e67e22", alpha=0.18,
           label=f"最大回撤区间 {dd.iloc[trough_i]:.1%}")
ax.annotate(f"最大回撤 {dd.iloc[trough_i]:.2%}",
            xy=(df["dt"].iloc[trough_i], df["nav"].iloc[trough_i]),
            xytext=(15, -28), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#e67e22"), fontsize=9)
ax.set_title("study_007 修复版: 样本外(2023-01~2025-12) 策略净值 vs 等权基准（对数轴）")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(RES / "nav_main_vs_benchmark.png", dpi=150)
plt.close(fig)
print("[chart] nav_main_vs_benchmark.png  最大回撤区间: "
      f"{df['trade_date'].iloc[peak_i]} -> {df['trade_date'].iloc[trough_i]} "
      f"({dd.iloc[trough_i]:.4%})")

# ---------------------------------------------------------------------------
# 图2: 回撤水下图
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 3.8))
bdd = df["bench_nav"] / df["bench_nav"].cummax() - 1.0
ax.fill_between(df["dt"], dd * 100, 0, color=C_STRAT, alpha=0.55, label="策略回撤")
ax.plot(df["dt"], bdd * 100, color=C_BENCH, lw=1.0, label="等权基准回撤")
ax.set_title("回撤曲线（水下图）")
ax.set_ylabel("回撤 (%)")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(RES / "drawdown_main.png", dpi=150)
plt.close(fig)
print("[chart] drawdown_main.png")

# ---------------------------------------------------------------------------
# 图3: 分年度收益柱状图（2023/2024/2025 样本外）
# ---------------------------------------------------------------------------
yr = pd.read_csv(RES / "annual_returns.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8.6, 4.6))
x = np.arange(len(yr))
w = 0.36
ax.bar(x - w / 2, yr["strategy"] * 100, w, color=C_STRAT, label="修复后策略")
ax.bar(x + w / 2, yr["benchmark"] * 100, w, color=C_BENCH, label="全A等权基准")
for i, (s, b) in enumerate(zip(yr["strategy"], yr["benchmark"])):
    ax.text(i - w / 2, s * 100 + 1, f"{s:.1%}", ha="center", fontsize=9)
    ax.text(i + w / 2, b * 100 + 1, f"{b:.1%}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"{i}年" for i in yr.index])
ax.set_ylabel("年度收益 (%)")
ax.set_title("样本外分年度收益对比（2023–2025）")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(RES / "annual_returns.png", dpi=150)
plt.close(fig)
print("[chart] annual_returns.png")

# ---------------------------------------------------------------------------
# 图4: 月度 Rank IC 柱状图
# ---------------------------------------------------------------------------
ic = load("main_monthly_ic.csv")
ic["dt"] = to_dt(ic["signal_date"])
colors = np.where(ic["ic"] >= 0, C_STRAT, "#2c3e50")
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(ic["dt"], ic["ic"], width=20, color=colors, alpha=0.85)
ax.axhline(ic["ic"].mean(), color="#e67e22", ls="--", lw=1.2,
           label=f"月均IC={ic['ic'].mean():.4f}")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("样本外月度 Rank IC（合成 score vs 次月收益）")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(RES / "monthly_ic.png", dpi=150)
plt.close(fig)
print(f"[chart] monthly_ic.png  IC>0: {(ic['ic']>0).mean():.4f}")

# ---------------------------------------------------------------------------
# 图5: Q1 小市值域 vs Q5 大市值域 净值对比
# ---------------------------------------------------------------------------
nv = load("size_domain_nav.csv")
nv["dt"] = to_dt(nv["trade_date"])
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(nv["dt"], nv["nav_q1"], color=C_Q1, lw=1.5, label="Q1 最小市值域 (top30)")
ax.plot(nv["dt"], nv["nav_q5"], color=C_Q5, lw=1.5, label="Q5 最大市值域 (top30)")
ax.plot(df["dt"], df["bench_nav"], color=C_BENCH, lw=1.1, ls="--",
        label="全A等权基准", alpha=0.8)
ax.set_yscale("log")
ax.set_title("分域净值对比: 真实流通市值五分位 Q1(小) vs Q5(大)，样本外 2023-01~2025-12（对数轴）")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(RES / "size_domains_nav.png", dpi=150)
plt.close(fig)
print("[chart] size_domains_nav.png")

print("全部图表完成。")
