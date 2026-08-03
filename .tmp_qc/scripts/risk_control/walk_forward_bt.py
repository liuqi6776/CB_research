# -*- coding: utf-8 -*-
"""
MA20 三档回撤控制 walk-forward / OOS 验证 (BASE+VAL+RS12 框架, 2020-2026)
三类验证:
  A. 子区间稳定性: 2020-2022 / 2023-2024 / 2025-2026 分段独立回测, MA20 三档 vs 无风控
  B. deep 参数敏感性: 0.95~1.00 扫描全样本 (检验是否孤峰过拟合 / 高原稳健)
  C. 滚动 walk-forward: 每年1月用截至当时的历史选 Sharpe 最优 deep, 应用到下一年
     (2020 为 warm-up 用默认 0.98), 对比固定 0.98 / 无风控
口径: 月度调仓 Top50, 20bps 双边, 风控降仓部分现金缓冲, RS12 弱时持 512100 ETF。

输出: results/walk_forward_bt.txt, results/walk_forward_nav.png
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic import style_factors as sf

OUT_DIR = rv.OUT_DIR
COST = rv.COST_BPS / 10000.0
TOP_N = rv.TOP_N
DEEPS = [0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
DEFAULT_DEEP = 0.98


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def stats(pr):
    """月收益序列 -> (年化, Sharpe, MaxDD, 卡玛)"""
    pr = pr.dropna()
    navs_c = (1 + pr).cumprod()
    years = len(pr) / 12.0
    cagr = navs_c.iloc[-1] ** (1 / years) - 1
    sharpe = pr.mean() / pr.std(ddof=1) * np.sqrt(12)
    mdd = ((navs_c.cummax() - navs_c) / navs_c.cummax()).max()
    calmar = cagr / mdd if mdd > 0 else np.nan
    return cagr, sharpe, mdd, calmar


def main():
    t0 = time.time()
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]
    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    val_map = sf.load_valuation(rebal, all_codes)
    funda_map = sf.build_funda_pit(rebal, all_codes)
    panels = sf.build_factors(val_map, funda_map, rebal)

    sml = load_idx("000852.SH")
    big = load_idx("000300.SH")
    etf = load_idx("512100.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    idx_ret = sml["pct_chg"] / 100.0
    idx_close = sml["close"]
    ma20 = idx_close.rolling(20).mean()
    # 修复同日信号前视: T-1 日收盘信号, T 日生效
    idx_close_1 = idx_close.shift(1)
    ma20_1 = ma20.shift(1)
    etf_ret = etf["pct_chg"] / 100.0

    # ---------- 每月 Top50 (BASE+VAL) ----------
    picks_map = {}
    for rb in rebal:
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        fvals = {}
        for code in members:
            f1, f2, ft = ret_1m.get(code), ivol.get(code), turn.get(code)
            fr = fwd.get(code)
            if fr is None or rb not in fr.index:
                continue
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            for name in panels:
                p = panels[name].get(rb)
                if p is not None and code in p.index:
                    v = p.loc[code]
                    if np.isfinite(v):
                        row[name] = v
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = sf.BASE_COLS + ["VAL"]
        has = zdf[cols].dropna()
        if len(has) < TOP_N:
            continue
        picks_map[rb] = has.mean(axis=1).nlargest(TOP_N).index.tolist()
    print(f"[load] Top50 月份 {len(picks_map)}, 耗时 {time.time()-t0:.0f}s", flush=True)

    # ---------- 对每个 deep 跑全样本日频风控 (deep=None 表示无风控) ----------
    def run_ma20(deep, rbl=None):
        """rbl: 子区间 rebal 列表(独立 NAV), 默认全样本"""
        rbl = rbl if rbl is not None else rebal
        nav_rb = {rbl[0]: 1.0}
        for i, rb in enumerate(rbl):
            if i + 1 >= len(rbl):
                continue
            rb_next = rbl[i + 1]
            if rb not in picks_map:
                nav_rb[rb_next] = nav_rb.get(rb, 1.0)
                continue
            hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
            hold = trade_dates[hi + 1:hn + 1]
            picks = picks_map[rb]
            comb = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
            comb_ret = comb.mean(axis=1)
            e_ret = etf_ret.reindex(hold).fillna(0.0)
            rs12_on = bool(sig_rs12.loc[rb]) if rb in sig_rs12.index else True
            nav = nav_rb.get(rb, 1.0)
            for t in hold:
                r_t = e_ret.loc[t]
                if rs12_on:
                    w = 1.0
                    if deep is not None:
                        c, m = idx_close_1.get(t, np.nan), ma20_1.get(t, np.nan)
                        if np.isfinite(c) and np.isfinite(m):
                            w = 1.0 if c >= m else (0.5 if c >= deep * m else 0.0)
                    r_t = w * comb_ret.loc[t]
                nav *= (1.0 + r_t)
            nav *= (1.0 - COST)
            nav_rb[rb_next] = nav
        return pd.Series(nav_rb).sort_index()

    navs_none = run_ma20(None)
    navs_deep = {d: run_ma20(d) for d in DEEPS}
    print(f"[bt] deep 扫描完成, 耗时 {time.time()-t0:.0f}s", flush=True)

    # ================= A. 子区间稳定性 =================
    segs = [
        ("2020-2022", "20200101", "20221231"),
        ("2023-2024", "20230101", "20241231"),
        ("2025-2026", "20250101", "20261231"),
    ]
    seg_rows = []
    for (sname, s0, s1) in segs:
        rbl = [rb for rb in rebal if s0 <= rb <= s1]
        if len(rbl) < 2:
            continue
        sn_none = run_ma20(None, rbl)
        sn_097 = run_ma20(0.97, rbl)
        sn_098 = run_ma20(0.98, rbl)
        pr_n = sn_none.pct_change().dropna()
        pr_7 = sn_097.pct_change().dropna()
        pr_8 = sn_098.pct_change().dropna()
        row = [sname]
        for tag, pr in (("无风控", pr_n), ("MA20.97", pr_7), ("MA20.98", pr_8)):
            cagr, sharpe, mdd, calmar = stats(pr)
            row += [f"{cagr:.1%}", f"{sharpe:.2f}", f"{mdd:.1%}", f"{calmar:.2f}"]
        seg_rows.append(row)

    # ================= B. deep 敏感性 (全样本) =================
    sens_rows = []
    for d in DEEPS:
        s = navs_deep[d]
        pr = s.pct_change().dropna()
        cagr, sharpe, mdd, calmar = stats(pr)
        ex = (1 + pr).prod() / navs_none.pct_change().dropna().pipe(
            lambda p: (1 + p).prod()) - 1
        sens_rows.append([d, cagr, sharpe, mdd, calmar, ex])
    pr_n = navs_none.pct_change().dropna()
    c_n = stats(pr_n)
    sens_rows.append([None] + list(c_n) + [0.0])

    # ================= C. 滚动 walk-forward =================
    # 每年1月重估: 用截至上年底的历史选 Sharpe 最优 deep (需>=12个月)
    sel = {}
    for y in range(2021, 2027):
        cut = f"{y}0101"
        past = [rb for rb in rebal if rb < cut]
        if len(past) < 12:
            continue
        best_d, best_sh = None, -np.inf
        for d in DEEPS:
            s = navs_deep[d].reindex(past).dropna()
            pr = s.pct_change().dropna()
            if len(pr) < 6:
                continue
            sh = pr.mean() / pr.std(ddof=1) * np.sqrt(12)
            if sh > best_sh:
                best_sh, best_d = sh, d
        sel[y] = (best_d, best_sh)

    # walk-forward NAV: 逐段用该段所选参数内部的相对收益复利拼接
    # (修复: 不再直接拼不同参数回测的绝对 NAV, 避免切换时把路径差异误算为收益)
    wf_nav = pd.Series(index=rebal, dtype=float)
    wf_nav[rebal[0]] = 1.0
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        rb_next = rebal[i + 1]
        # 收益期 (rb, rb_next] 归 rb_next 所属年份选参 (修复: 参数晚一个持有期生效)
        y = int(rb_next[:4])
        if y == 2020:
            d = DEFAULT_DEEP
        else:
            d = sel.get(y, (DEFAULT_DEEP, None))[0]
        a, b = navs_deep[d].get(rb), navs_deep[d].get(rb_next)
        if a and b:
            wf_nav[rb_next] = wf_nav[rb] * (b / a)

    pr_wf = wf_nav.pct_change().dropna()
    pr_fix = navs_deep[DEFAULT_DEEP].pct_change().dropna()
    wf_stats = stats(pr_wf)
    fix_stats = stats(pr_fix)
    ex_wf = (1 + pr_wf).prod() / (1 + pr_n.reindex(pr_wf.index)).prod() - 1
    ex_fix = (1 + pr_fix).prod() / (1 + pr_n.reindex(pr_fix.index)).prod() - 1

    # ================= 输出 =================
    print("\n" + "=" * 110)
    print("MA20 三档 walk-forward / OOS 验证 (BASE+VAL+RS12, 月度调仓 Top50, 20bps, 2020-2026)")
    print("=" * 110)

    print("\n[A] 子区间独立回测 (每年卡玛列)")
    hdr = f"{'子区间':<10}{'策略':>10}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'卡玛':>7}"
    print(hdr)
    out_lines = [hdr]
    for row in seg_rows:
        sname = row[0]
        for j in range(3):
            base = 1 + j * 4
            tag = ["无风控", "MA20.97", "MA20.98"][j]
            line = f"{sname if j == 0 else '':<10}{tag:>10}{row[base]:>8}{row[base+1]:>8}{row[base+2]:>9}{row[base+3]:>7}"
            print(line)
            out_lines.append(line)

    print("\n[B] deep 敏感性 (全样本)")
    hdr = f"{'deep':>6}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'卡玛':>7}{'超额v无风控':>12}"
    print(hdr)
    out_lines.append("\n[B] deep 敏感性 (全样本)")
    out_lines.append(hdr)
    for r in sens_rows:
        d, cagr, sharpe, mdd, calmar, ex = r
        dl = "无风控" if d is None else f"{d:.2f}"
        line = f"{dl:>6}{cagr:>8.2%}{sharpe:>8.2f}{mdd:>9.2%}{calmar:>7.2f}{ex:>12.2%}"
        print(line)
        out_lines.append(line)

    print("\n[C] walk-forward 每年选参 (截至上年底 Sharpe 最优)")
    hdr = f"{'重估时点':<12}{'选用deep':>10}{'当时Sharpe':>12}"
    print(hdr)
    out_lines.append("\n[C] walk-forward 每年选参")
    out_lines.append(hdr)
    for y in sorted(sel):
        d, sh = sel[y]
        line = f"{y}-01 (用{y-1}前数据):{d:>10.2f}{sh:>12.2f}"
        print(line)
        out_lines.append(line)

    print("\n[D] walk-forward 全样本对比")
    hdr = f"{'策略':<18}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'月胜率':>8}{'卡玛':>7}{'超额v无风控':>12}"
    print(hdr)
    out_lines.append("\n[D] walk-forward 全样本对比")
    out_lines.append(hdr)
    for tag, pr, ex in (("无风控", pr_n, 0.0),
                        ("固定MA20(0.98)", pr_fix, ex_fix),
                        ("walk-forward", pr_wf, ex_wf)):
        cagr, sharpe, mdd, calmar = stats(pr)
        line = f"{tag:<18}{cagr:>8.2%}{sharpe:>8.2f}{mdd:>9.2%}{(pr>0).mean():>8.1%}{calmar:>7.2f}{ex:>12.2%}"
        print(line)
        out_lines.append(line)

    with open(os.path.join(OUT_DIR, "walk_forward_bt.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")

    # ---------- 图 ----------
    # ETF 对照: 完整月收益 (rb, rb_next] 复利记在 rb_next, 与策略 NAV 对齐
    # (修复: 原取调仓日单日收益, 视觉对照偏低/失真)
    etf_m = {}
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        rb_next = rebal[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        etf_m[rb_next] = (1 + etf_ret.reindex(trade_dates[hi + 1:hn + 1]).fillna(0.0)).prod() - 1
    etf_m = pd.Series(etf_m)
    fig, ax = plt.subplots(figsize=(13, 7.5))
    x_all = sorted(rebal)
    ax.plot(np.arange(len(x_all)), navs_none.reindex(x_all).ffill().values,
            label="无风控 BASE+VAL", lw=1.2, ls=":", color="#888")
    ax.plot(np.arange(len(x_all)), navs_deep[DEFAULT_DEEP].reindex(x_all).ffill().values,
            label=f"固定 MA20(0.98)", lw=1.8, color="#c33")
    ax.plot(np.arange(len(x_all)), wf_nav.reindex(x_all).ffill().values,
            label="walk-forward (每年滚动选参)", lw=1.8, color="#26c")
    ax.plot(np.arange(len(x_all)), (1 + etf_m.reindex(x_all).fillna(0)).cumprod().values,
            label="512100ETF", lw=1.2, ls="--", color="#999")
    ax.set_yscale("log")
    ax.set_ylabel("净值(对数)")
    ax.set_title("MA20 三档 walk-forward 验证 (BASE+VAL+RS12, 2020-2026)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    # deep 选择标注
    for y in sorted(sel):
        d, _ = sel[y]
        rb0 = f"{y}0101"
        if rb0 in x_all:
            i = x_all.index(rb0)
            ax.annotate(f"{d:.2f}", (i, wf_nav[rb0]), fontsize=8, color="#26c",
                        xytext=(0, 8), textcoords="offset points", ha="center")
    fp = os.path.join(OUT_DIR, "walk_forward_nav.png")
    plt.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[保存图] {fp}   (总耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
