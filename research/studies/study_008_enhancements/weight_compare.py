# -*- coding: utf-8 -*-
"""权重口径对照: ≥0.93 等权 vs Top60 IVW120 (同口径叠加阶段5 集中度)

变体:
  A. Top60基线     : IVW120 + 阶段4 + 阶段5 (v1.1.0-wan1, 控制变量)
  B. >=0.93 等权   : score>=0.93 全选等权 + 阶段4 (无集中度)   <- 细扫峰值
  C. >=0.93 等权+集中度: B + 阶段5 集中度 (与 A 完全同口径)

目的: 排除"权重方式"与"集中度"两个变量, 验证等权是否真优于 IVW120.

注: 等权在持仓 <25 只时 1/N > 4% 单股 cap 数学不可行 (clip->归一化震荡),
    该类月份 C 的集中度约束退化, 报告中用 n_selected 量化.
"""
import json
import os
import sys

import numpy as np

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

THR = 0.93


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

    def _run(name, use_hrp, with_conc):
        s, st = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                               e_ovn, e_intra, use_hrp=use_hrp, use_ma20=False,
                               st_map=st_map, limit_sets=(one_up, one_dn),
                               tradable=tf5, concentration=_conc if with_conc else None,
                               score_thr=None if use_hrp else THR)
        return s, st

    runs = {}
    runs["A Top60基线 (IVW120+集中度)"] = _run("A", use_hrp=True, with_conc=True)
    runs["B >=0.93 等权 (无集中度)"] = _run("B", use_hrp=False, with_conc=False)
    runs["C >=0.93 等权+集中度"] = _run("C", use_hrp=False, with_conc=True)

    lines = ["权重口径对照 (A 基线 vs B 等权 vs C 等权+集中度, 万1)", "=" * 100]
    hdr = f"{'变体':<26}{'终值':>9}{'年化':>9}{'Sharpe':>8}{'MaxDD':>9}{'卡玛':>7}" \
          f"{'月均':>8}{'胜率':>7}{'最差月':>9}{'持仓均值':>9}"
    lines.append(hdr)
    metrics = {}
    for name, (s, st) in runs.items():
        m = _metrics(s)
        metrics[name] = m
        ns = st.get("n_selected") or []
        n_s = f"{np.mean(ns):>8.1f}" if ns else f"{'60(固定)':>8}"
        lines.append(f"{name:<26}{m['final']:>9.4f}{m['ann']:>9.2%}{m['sharpe']:>8.2f}"
                     f"{m['mdd']:>9.2%}{m['calmar']:>7.2f}{m['m_mean']:>8.2%}"
                     f"{m['m_win']:>7.0%}{m['m_worst']:>9.2%}{n_s}")

    # 低持仓期 (等权时 1/N > 4% 单股 cap 数学不可行) 统计
    ns = runs["B >=0.93 等权 (无集中度)"][1].get("n_selected") or []
    arr = np.asarray(ns, dtype=float)
    n_unfeas = int((arr < 25).sum())
    lines.append("")
    lines.append(f">=0.93 等权: 持仓 <25 只 (1/N>4% 单股 cap 数学不可行) 共 {n_unfeas}/{len(arr)} 月, "
                 f"min {int(arr.min())}")
    lines.append("  说明: C 变体在这些月份单股 4% cap 无法满足, clip-归一化迭代震荡, 约束退化; "
                 "结果中该类月份仅受行业/容量约束.")
    for name, (s, st) in runs.items():
        lines.append(f"  {name}: n_missing {st['n_missing']} 月 | 剔除 {st['n_trad_removed']} 只次 | "
                     f"买阻塞 {st['n_buy_block']} | 停牌 {st['n_suspend']}")
    print("\n".join(lines))

    # ---- 净值图 ----
    dts = pd_to_datetime(runs["A Top60基线 (IVW120+集中度)"][0].index)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    styles = [("#333", 1.8), ("#1f77b4", 1.4), ("#d62728", 1.4)]
    for (name, (s, st)), (c, lw) in zip(runs.items(), styles):
        ax.plot(dts, s.values, lw=lw, color=c,
                label=f"{name} (年化 {metrics[name]['ann']:.1%}, 卡玛 {metrics[name]['calmar']:.2f})")
    ax.set_title("≥0.93 等权 vs Top60 IVW120 基线 — 同口径叠加阶段5 集中度对照")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp1 = os.path.join(C.OUT_DIR, "weight_compare_nav.png")
    fig.savefig(fp1, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {fp1}")

    # ---- 落盘 ----
    out = {name: dict(metrics=metrics[name],
                      n_selected=[int(v) for v in (st.get("n_selected") or [])],
                      n_missing=int(st["n_missing"]), n_trad_removed=int(st["n_trad_removed"]))
           for name, (s, st) in runs.items()}
    out["_meta"] = dict(n_unfeas_months=n_unfeas,
                        note="持仓<25只时等权1/N>4%单股cap数学不可行, 该月仅行业/容量约束生效")
    with open(os.path.join(C.OUT_DIR, "weight_compare.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(C.OUT_DIR, "weight_compare.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'weight_compare.txt')} | .json")


def pd_to_datetime(idx):
    import pandas as pd
    return pd.to_datetime(idx, format="%Y%m%d")


if __name__ == "__main__":
    main()
