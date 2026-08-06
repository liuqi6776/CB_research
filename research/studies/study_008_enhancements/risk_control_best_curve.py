# -*- coding: utf-8 -*-
"""一次性: 导出最佳策略 vs 推荐配置 vs 中证1000 月度净值 (供对话内收益曲线, 与 baseline_cmp 同口径)"""
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.risk_control_real import TIER5

CASES = [
    ("HRP纯选股", dict(use_hrp=True, use_ma20=False)),
    ("推荐配置", dict(use_hrp=True, use_ma20=True, tier=TIER5,
                     dd_stop=0.10, dd_floor=0.15, stop_w=0.5, floor_w=0.5, recov=0.05)),
]


def month_end(s):
    s2 = s.copy()
    s2.index = pd.to_datetime(s2.index.astype(str), format="%Y%m%d")
    return s2.resample("M").last()


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)

    out = {}
    for lb, kw in CASES:
        s, _ = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                              e_ovn, e_intra, **kw)
        out[lb] = s
        print(f"[backtest] {lb}", flush=True)

    # 中证1000 指数 (升序), 与策略同一起点归一 (策略净值起点=1)
    start = str(out["HRP纯选股"].index[0])
    idx = pd.read_parquet(os.path.join(rv.IDX_DIR, "000852.SH.parquet"))
    ix = idx.set_index("trade_date")["close"].astype(float).sort_index()
    ix.index = ix.index.astype(str)
    ix = ix[ix.index >= start]
    out["中证1000"] = ix / ix.loc[start]

    # 月度抽样 + 对齐
    out = {lb: month_end(s) for lb, s in out.items()}
    common = out["HRP纯选股"].index.intersection(out["中证1000"].index)
    rows = {d.strftime("%Y-%m"): {} for d in common}
    for lb, s in out.items():
        s = s.reindex(common)
        for d, v in s.items():
            rows[d.strftime("%Y-%m")][lb] = round(float(v), 4)

    fp = os.path.join(C.OUT_DIR, "risk_control_best_curve.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=0)
    print(f"[saved] {fp}  (月度 {len(rows)} 点)")
    for lb in ["HRP纯选股", "推荐配置", "中证1000"]:
        vals = [v[lb] for v in rows.values()]
        print(f"  {lb}: 首月末 {vals[0]} 终点 {vals[-1]} 累计 {vals[-1]-1:+.1%}")


def plot_only():
    """python risk_control_best_curve.py plot: 仅从 JSON 重绘 PNG"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fp = os.path.join(C.OUT_DIR, "risk_control_best_curve.json")
    with open(fp, encoding="utf-8") as f:
        rows = json.load(f)

    keys = list(rows.keys())
    x = pd.to_datetime([k + "-28" for k in keys])
    series = {"HRP纯选股": "#3C2ECA", "推荐配置": "#6F6FFF", "中证1000": "#A9AEFF"}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=110)
    oos = pd.to_datetime("2023-01-28")
    ax.axvspan(oos, x[-1], color="#D3D4DA", alpha=0.28, zorder=0)
    ax.axvline(oos, color="#52525B", linestyle="--", lw=1, alpha=0.6, zorder=1)

    ymax = max(max(r.values()) for r in rows.values())
    ax.text(oos + pd.Timedelta(days=8), ymax * 0.985, "OOS段 2023起",
            fontsize=11, color="#52525B", va="top")

    for lb, color in series.items():
        vals = [r[lb] for r in rows.values()]
        kw = dict(color=color, lw=2.8 if lb == "HRP纯选股" else 2.0)
        if lb == "中证1000":
            kw = dict(color=color, lw=1.4, ls="--", alpha=0.85)
        ax.plot(x, vals, label=f"{lb}  (累计 {vals[-1]-1:+.1%})", **kw)

    ax.set_title("月度净值曲线 2020–2026（策略与中证1000 同起点=1 · 月频）", fontsize=14)
    ax.set_ylabel("净值（起点=1）", fontsize=11)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = os.path.join(C.OUT_DIR, "risk_control_best_curve.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[saved] {out}")


def svg_data():
    """python risk_control_best_curve.py svg: 输出静态 SVG 折线坐标"""
    fp = os.path.join(C.OUT_DIR, "risk_control_best_curve.json")
    with open(fp, encoding="utf-8") as f:
        rows = json.load(f)
    keys = list(rows.keys())
    n = len(keys)
    X0, X1, Y0, Y1, VMIN, VMAX = 64, 700, 24, 326, 0.85, 2.45

    def x(i):
        return X0 + i * (X1 - X0) / (n - 1)

    def y(v):
        return Y1 - (v - VMIN) / (VMAX - VMIN) * (Y1 - Y0)

    for lb in ["HRP纯选股", "推荐配置", "中证1000"]:
        pts = " ".join(f"{x(i):.1f},{y(r[lb]):.1f}" for i, r in enumerate(rows.values()))
        print(f"[poly {lb}] {pts}")
    oos_i = keys.index("2023-01")
    print(f"[oos_x] {x(oos_i):.1f}")
    for v in [1.0, 1.5, 2.0, 2.4]:
        print(f"[grid] {v:.2f} {y(v):.1f}")
    for mi, mk in [(12, "2021-02"), (36, "2023-02"), (60, "2025-02")]:
        print(f"[xtick] {x(mi):.1f} {mk}")
    print(f"[xlast] {x(n - 1):.1f} {keys[-1]}")


def metrics():
    """python risk_control_best_curve.py metrics: 按图内月频数据计算 累计/年化/MaxDD/波动/Sharpe/卡玛"""
    fp = os.path.join(C.OUT_DIR, "risk_control_best_curve.json")
    with open(fp, encoding="utf-8") as f:
        rows = json.load(f)
    keys = list(rows.keys())
    n = len(keys) - 1  # 月区间数
    print(f"{'口径':<20}{'累计':>9}{'年化':>9}{'MaxDD':>9}{'年化波动':>9}{'Sharpe':>8}{'卡玛':>8}")
    for lb in ["HRP纯选股", "推荐配置", "中证1000"]:
        nav = [r[lb] for r in rows.values()]
        cum = nav[-1] - 1
        cagr = nav[-1] ** (12 / n) - 1
        peak = nav[0]
        mdd = 0.0
        for v in nav:
            peak = max(peak, v)
            mdd = min(mdd, v / peak - 1)
        rets = [nav[i] / nav[i - 1] - 1 for i in range(1, len(nav))]
        import statistics
        mean_r = statistics.mean(rets)
        std_r = statistics.pstdev(rets)
        vol = std_r * (12 ** 0.5)
        sharpe = mean_r / std_r * (12 ** 0.5) if std_r else float("nan")
        calmar = cagr / abs(mdd) if mdd else float("nan")
        print(f"{lb:<20}{cum:>8.1%}{cagr:>8.2%}{mdd:>8.2%}{vol:>8.2%}{sharpe:>8.2f}{calmar:>8.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_only()
    elif len(sys.argv) > 1 and sys.argv[1] == "svg":
        svg_data()
    elif len(sys.argv) > 1 and sys.argv[1] == "metrics":
        metrics()
    else:
        main()
