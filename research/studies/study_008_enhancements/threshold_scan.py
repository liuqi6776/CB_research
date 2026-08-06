# -*- coding: utf-8 -*-
"""分数阈值选股对照 (vs 固定 Top60, v1.1.0-wan1 口径)

实验: 把"截面 Top60 固定数量"改为"等权买入全部 score >= thr 的成分股",
     看持仓数量与收益分布随阈值的变化 (用户提出 0.65 / 0.8 / 0.9 等档位).

口径:
- 基线对照: v1.1.0-wan1 (Top60 + IVW120 + 阶段4 过滤 + 阶段5 集中度, 万1)
- 阈值变体: score >= thr 全选 + 等权 (use_hrp=False) + 阶段4 可交易过滤
  (等权天然分散; 且低数量期单股 4% cap 数学不可行 → 阈值变体不带阶段5 集中度)
- RS12 择时保留 (弱时持 512100 ETF); fail-closed 沿用上期 (订单名单 <10 只或打分缺失)
- 打分 = zscore 均值 (ret_1m + ivol + turnover_vol_20 + VAL, 缺 VAL 降级 BASE 3 因子)

输出:
  results/threshold_scan.txt / .json
  results/threshold_scan_nav.png     净值曲线 (基线 + 各阈值)
  results/threshold_scan_ndist.png   每期持仓数量分布 (箱线)
  results/threshold_scan_monthly.png 月度收益分布 (箱线)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.concentration import apply_concentration, amount60_at
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df

THRESHOLDS = [0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.98]


def _metrics(s):
    n = len(s)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
    ret = s.pct_change().dropna()
    shp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(242.0)
    dd = ((s.cummax() - s) / s.cummax()).max()
    m_ret = s.groupby(s.index.str[:6]).last().pct_change().dropna()
    return dict(final=float(s.iloc[-1]), ann=cagr, sharpe=shp, mdd=float(dd),
                calmar=float(cagr / dd) if dd > 0 else 0.0,
                m_mean=float(m_ret.mean()), m_win=float((m_ret > 0).mean()),
                m_worst=float(m_ret.min()), m_best=float(m_ret.max()))


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)
    st_map = E.load_st_intervals()
    one_up, one_dn = E.build_limit_sets(open_df, high_df, low_df, pct_df, env.all_codes)
    amount_df = load_amount_df(env, td)
    ind_map = C.load_industry_map()
    tf5 = Tradability(td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                      st_map=st_map, min_vol=12.0, pct_df=pct_df)

    def _conc(rb, w, nav_pre):
        return apply_concentration(w, ind_map=ind_map, cap_stock=0.04, cap_ind=0.20,
                                   cap_top5=0.20,
                                   amount60=amount60_at(amount_df, td, rb),
                                   nav_pre=nav_pre, cap_amount=0.05, scale=1e8)

    runs = {}
    # 基线: v1.1.0-wan1 (Top60 + IVW120 + 阶段4/5, 万1)
    s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                           e_ovn, e_intra, use_hrp=True, use_ma20=False,
                           st_map=st_map, limit_sets=(one_up, one_dn),
                           tradable=tf5, concentration=_conc)
    runs["Top60基线"] = dict(nav=s, stats=st, n_sel=None)
    # 阈值变体: score >= thr 等权 + 阶段4 过滤 (无集中度)
    for thr in THRESHOLDS:
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=False, use_ma20=False,
                               st_map=st_map, limit_sets=(one_up, one_dn),
                               tradable=tf5, score_thr=thr)
        runs[f">={thr}"] = dict(nav=s, stats=st, n_sel=st.get("n_selected", []))

    # ---- 指标表 ----
    lines = ["分数阈值选股对照 (等权买入 score>=thr, vs 固定 Top60 基线)", "=" * 96]
    hdr = f"{'变体':<12}{'终值':>9}{'年化':>9}{'Sharpe':>8}{'MaxDD':>9}{'卡玛':>7}" \
          f"{'月均':>8}{'胜率':>7}{'最差月':>9}{'持仓均值':>9}{'[min,max]':>12}"
    lines.append(hdr)
    metrics = {}
    for name, r in runs.items():
        m = _metrics(r["nav"])
        metrics[name] = m
        ns = r["n_sel"]
        if ns:
            n_mean = np.mean(ns)
            n_lo, n_hi = min(ns), max(ns)
            n_s = f"{n_mean:>8.1f}{f'[{n_lo},{n_hi}]':>12}"
        else:
            n_s = f"{'60(固定)':>8}{'':>12}"
        lines.append(f"{name:<12}{m['final']:>9.4f}{m['ann']:>9.2%}{m['sharpe']:>8.2f}"
                     f"{m['mdd']:>9.2%}{m['calmar']:>7.2f}{m['m_mean']:>8.2%}"
                     f"{m['m_win']:>7.0%}{m['m_worst']:>9.2%}{n_s}")

    lines.append("")
    lines.append("持仓数量与 fail-closed 明细:")
    for name, r in runs.items():
        st = r["stats"]
        ns = r["n_sel"]
        if ns is None:
            lines.append(f"  Top60基线: 固定 60 只/期 (阶段4 剔除 {st['n_trad_removed']} 只次), "
                         f"fail-closed {st['n_missing']} 月")
        else:
            arr = np.asarray(ns, dtype=float)
            fc10 = sum(1 for v in ns if 0 < v < 10)
            nz0 = sum(1 for v in ns if v == 0)
            lines.append(f"  >= {name[1:]}: 持仓 均值 {arr.mean():.1f} | 中位 {np.median(arr):.0f} | "
                         f"min {int(arr.min())} | max {int(arr.max())} | "
                         f"0只月 {nz0} | 1-9只月 {fc10} | fail-closed {st['n_missing']} 月 | "
                         f"剔除 {st['n_trad_removed']} 只次")
    print("\n".join(lines))

    # ---- 图1: 净值 (代表档, 避免 12 条线混乱) ----
    dts = pd.to_datetime(runs["Top60基线"]["nav"].index, format="%Y%m%d")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(dts, runs["Top60基线"]["nav"].values, label="Top60 基线 (IVW120+集中度)", lw=1.8, color="#333")
    cmap = plt.cm.viridis
    for t in [0.85, 0.89, 0.93, 0.95, 0.98]:
        name = f">={t}"
        if name not in runs:
            continue
        m = metrics[name]
        c = cmap(0.15 + 0.7 * (THRESHOLDS.index(t) / max(1, len(THRESHOLDS) - 1)))
        ax.plot(dts, runs[name]["nav"].values, lw=1.3, color=c,
                label=f">={t} 等权 (年化 {m['ann']:.1%}, 持仓均值 {np.mean(runs[name]['n_sel']):.0f})")
    ax.set_title("分数阈值选股 vs 固定 Top60 基线 (等权, 阶段4 过滤, RS12 择时, 万1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp1 = os.path.join(C.OUT_DIR, "threshold_scan_nav.png")
    fig.savefig(fp1, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {fp1}")

    # ---- 图1b: 阈值-绩效分布曲线 (哪个阈值最优) ----
    xs = THRESHOLDS
    anns = [metrics[f">={t}"]["ann"] for t in xs]
    shps = [metrics[f">={t}"]["sharpe"] for t in xs]
    cals = [metrics[f">={t}"]["calmar"] for t in xs]
    mdds = [metrics[f">={t}"]["mdd"] for t in xs]
    nmeans = [np.mean(runs[f">={t}"]["n_sel"]) for t in xs]
    b_ann, b_shp, b_cal, b_mdd = (metrics["Top60基线"][k] for k in ("ann", "sharpe", "calmar", "mdd"))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    spec = [
        (axes[0][0], "年化收益", anns, b_ann, False, True),
        (axes[0][1], "Sharpe", shps, b_shp, False, False),
        (axes[1][0], "卡玛 (年化/MaxDD)", cals, b_cal, False, False),
        (axes[1][1], "MaxDD (深, 越低越好)", mdds, b_mdd, True, True),
    ]
    for (ax, title, ys, base, use_min, pct) in spec:
        ax.plot(xs, ys, marker="o", color="#1f77b4", lw=1.6)
        ax.axhline(base, color="#333", ls="--", lw=1.1,
                   label=f"Top60基线 {base:.2%}" if pct else f"Top60基线 {base:.2f}")
        ax.set_title(title)
        ax.set_xlabel("分数阈值 (score ≥ thr)")
        ax.grid(alpha=0.3)
        best = min(ys) if use_min else max(ys)
        bi = ys.index(best)
        ax.scatter([xs[bi]], [best], s=60, facecolor="none", edgecolor="#d62728", zorder=5)
        ax.annotate(f"最优 {xs[bi]}: {best:.2%}" if pct else f"最优 {xs[bi]}: {best:.2f}",
                    (xs[bi], best), textcoords="offset points", xytext=(8, 6),
                    fontsize=8, color="#d62728")
        ax.legend(fontsize=8)
    ax0 = axes[0][0].twinx()
    ax0.plot(xs, nmeans, marker="s", color="#d62728", lw=1.2, alpha=0.75, ls="--", label="持仓均值(只)")
    ax0.set_ylabel("持仓均值(只)", color="#d62728")
    ax0.tick_params(axis="y", labelcolor="#d62728")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("分数阈值-绩效分布曲线 (等权, 0.0~0.99 步 0.1; 虚线=Top60 基线)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp1b = os.path.join(C.OUT_DIR, "threshold_curve.png")
    fig.savefig(fp1b, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp1b}")

    # ---- 图2: 持仓数量分布 ----
    names = [k for k in runs if runs[k]["n_sel"] is not None]
    colors = [plt.cm.viridis(i / max(1, len(names) - 1)) for i in range(len(names))]
    data = [runs[n]["n_sel"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(data, labels=[n for n in names], patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors[:len(names)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_title("每期持仓数量分布 (score>=thr 全选等权)")
    ax.set_ylabel("持仓只数")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fp2 = os.path.join(C.OUT_DIR, "threshold_scan_ndist.png")
    fig.savefig(fp2, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp2}")

    # ---- 图3: 月度收益分布 ----
    mr_data = [_metrics(runs[n]["nav"]) for n in ["Top60基线"] + names]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    monthly = []
    for n in ["Top60基线"] + names:
        s = runs[n]["nav"]
        monthly.append(s.groupby(s.index.str[:6]).last().pct_change().dropna().values)
    bp = ax.boxplot(monthly, labels=["Top60基线"] + names, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#333"] + colors[:len(names)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_title("月度收益分布 (箱线, 全段)")
    ax.set_ylabel("月度收益")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fp3 = os.path.join(C.OUT_DIR, "threshold_scan_monthly.png")
    fig.savefig(fp3, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp3}")

    # ---- 落盘 ----
    out = {}
    for name, r in runs.items():
        st = r["stats"]
        ns = r["n_sel"]
        out[name] = dict(metrics=metrics[name],
                         n_selected=[int(v) for v in ns] if ns is not None else None,
                         n_missing=int(st["n_missing"]), n_trad_removed=int(st["n_trad_removed"]),
                         n_buy_block=int(st["n_buy_block"]), n_suspend=int(st["n_suspend"]))
    with open(os.path.join(C.OUT_DIR, "threshold_scan.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(C.OUT_DIR, "threshold_scan.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'threshold_scan.txt')} | .json")


if __name__ == "__main__":
    main()
