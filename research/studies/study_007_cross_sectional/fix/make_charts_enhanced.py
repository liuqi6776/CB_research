# -*- coding: utf-8 -*-
"""
make_charts_enhanced.py — results_enhanced 图表生成（托管 python，matplotlib 可用）
读 results_enhanced/*.csv，输出 4 张 PNG 到 results_enhanced/。
运行: python make_charts_enhanced.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIX = Path(__file__).resolve().parent
OUT = FIX / "results_enhanced"

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot  # noqa: E402
setup_plot()

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

A_TAGS = ["A0", "A1", "A2", "A3", "A4"]
A_LABEL = {"A0": "A0 基线等权无过滤", "A1": "A1 剔北交所",
           "A2": "A2 剔BJ+剔最小20%", "A3": "A3 A2+inv_vol", "A4": "A4 A2+score加权"}


def main():
    navs = pd.read_csv(OUT / "navs_all.csv", dtype={"trade_date": str})
    navs["dt"] = pd.to_datetime(navs["trade_date"], format="%Y%m%d")

    # ---- 图1: A组净值对比（对数轴）----
    fig, ax = plt.subplots(figsize=(11, 6))
    for t in A_TAGS:
        ax.plot(navs["dt"], navs[t], label=A_LABEL[t], lw=1.4)
    ax.plot(navs["dt"], navs["BENCH"], label="全A等权基准", lw=1.2,
            color="gray", ls="--")
    ax.set_yscale("log")
    ax.set_title("A组变体净值对比（2023-01 ~ 2025-12，对数轴）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "nav_group_A_log.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] nav_group_A_log.png")

    # ---- 图2: A组分年度超额收益柱状图 ----
    ex = pd.read_csv(OUT / "annual_excess_A.csv", dtype={"year": str})
    years = ["2023", "2024", "2025"]
    x = np.arange(len(years))
    w = 0.15
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, t in enumerate(A_TAGS):
        v = ex[ex["variant"] == t].set_index("year").loc[years, "excess"].to_numpy()
        ax.bar(x + (i - 2) * w, v * 100, w, label=A_LABEL[t])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("超额收益 (%)")
    ax.set_title("A组变体分年度超额收益（策略 − 全A等权基准）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "annual_excess_A.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] annual_excess_A.png")

    # ---- 图3: B组 LS 净值曲线 ----
    fig, ax = plt.subplots(figsize=(11, 6))
    for t, lab in [("B1", "B1 全宇宙LS 50/50"), ("B2", "B2 剔BJ+剔小市值LS 50/50")]:
        d = pd.read_csv(OUT / f"nav_{t}.csv", dtype={"trade_date": str})
        d["dt"] = pd.to_datetime(d["trade_date"], format="%Y%m%d")
        ax.plot(d["dt"], d["nav"], label=lab, lw=1.4)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.set_title("B组多空净值（美元中性 50/50，cost=0.003；研究性检验，A股个股不可做空）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "nav_group_B_ls.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] nav_group_B_ls.png")

    # ---- 图4: 回撤对比水下图（A0/A2/A3/B1）----
    fig, ax = plt.subplots(figsize=(11, 6))
    for t, lab in [("A0", A_LABEL["A0"]), ("A2", A_LABEL["A2"]),
                   ("A3", A_LABEL["A3"]), ("B1", "B1 全宇宙LS")]:
        d = navs[["dt", t]].dropna() if t in navs.columns else None
        dd = d[t] / d[t].cummax() - 1
        ax.plot(d["dt"], dd * 100, label=lab, lw=1.2)
    ax.set_title("回撤对比水下图（A0 / A2 / A3 / B1）")
    ax.set_ylabel("回撤 (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "drawdown_underwater.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] drawdown_underwater.png")


if __name__ == "__main__":
    main()
