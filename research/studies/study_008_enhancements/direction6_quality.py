# -*- coding: utf-8 -*-
"""方向C: 质量因子 QMJ (ROE/毛利率/净利增速) 加入选股 — 2023+ 子区间
- 数据: D:/iquant_data/data_v2/fundamental1/fina_indicator_cache.parquet (2023-04 起, ann_date 无前视)
- 质量分: 调仓日取 ann_date<=rb 的最新一期, 横截面 zscore(roe)+zscore(grossprofit_margin)+zscore(netprofit_yoy), 缺失补 0
- 子区间: 2023-07~2026-06 (首期可用公告 2023-04)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.factor_dic import run_validation as rv
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements import common as C

FIN_FP = "D:/iquant_data/data_v2/fundamental1/fina_indicator_cache.parquet"
SUB_START = "20230630"  # 子区间起点
MA20_DEEP = 0.98
Q_COLS = ["roe", "grossprofit_margin", "netprofit_yoy"]


def load_fina():
    df = pd.read_parquet(FIN_FP)
    df["ts_code"] = df["ts_code"].astype(str)
    df["ann_date"] = df["ann_date"].astype(str)
    df["end_date"] = df["end_date"].astype(str)
    for c in Q_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 按 股票+公告日 排序, 保留最新一期
    df = df.sort_values(["ts_code", "ann_date"]).groupby("ts_code").tail(1)  # 避免重复 ann 取最后
    df = df.drop_duplicates(subset=["ts_code", "ann_date"], keep="last")
    return df


def quality_map(env, fina, rb):
    """调仓日 rb: 每只成分股的质量分 (缺失补0)"""
    members = rv.load_index_weight(rb)
    if members is None:
        return None
    latest = fina[fina["ann_date"] <= rb]
    latest = latest.sort_values("ann_date").groupby("ts_code").tail(1)
    latest = latest[latest["ts_code"].isin(members)]
    if latest.empty:
        return None
    sub = latest.set_index("ts_code")[Q_COLS]
    # 横截面 winsorize + zscore
    z = sub.apply(sf.winsorize_series).apply(lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
    q = z.sum(axis=1, min_count=1)  # 3 列全缺失 -> NaN
    # 合并到所有成员, 缺失补 0 (中性)
    out = pd.Series(0.0, index=sorted(members))
    out.update(q.dropna())
    return out


def build_picks(env, q_map, use_q):
    picks_map = {}
    for rb in env.rebal:
        if rb < SUB_START:
            continue
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        fvals = {}
        for code in members:
            f1, f2, ft = env.ret_1m.get(code), env.ivol.get(code), env.turn.get(code)
            fr = env.fwd.get(code)
            if fr is None or rb not in fr.index:
                continue
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            for name in env.panels:
                p = env.panels[name].get(rb)
                if p is not None and code in p.index:
                    v = p.loc[code]
                    if np.isfinite(v):
                        row[name] = v
            if use_q:
                qs = q_map.get(rb)
                if qs is not None and code in qs.index:
                    row["q"] = qs.loc[code]
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < C.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = sf.BASE_COLS + ["VAL"] + (["q"] if use_q else [])
        cols = [c for c in cols if c in zdf.columns]
        has = zdf[cols].dropna()
        if len(has) < C.TOP_N:
            continue
        picks_map[rb] = has.mean(axis=1).nlargest(C.TOP_N).index.tolist()
    return picks_map


def backtest_sub(env, picks_map, use_ma20=True):
    navs = {}
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if rb < SUB_START:
            continue
        nav = navs.get(rb, 1.0)
        if picks_map.get(rb) is None:
            navs[rb_next] = nav
            continue
        picks = picks_map[rb]
        comb = env.pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
        w = pd.Series(1.0 / len(picks), index=picks)
        cr = (comb * w.reindex(comb.columns)).sum(axis=1, min_count=1)
        for t in hold:
            r_t = e_ret.loc[t]
            if rs12_on:
                ww = 1.0
                if use_ma20:
                    c = env.idx_close_1.get(t, np.nan)
                    m = env.ma20_1.get(t, np.nan)
                    if np.isfinite(c) and np.isfinite(m):
                        ww = 1.0 if c >= m else (0.5 if c >= MA20_DEEP * m else 0.0)
                r_t = ww * cr.loc[t]
            nav *= (1.0 + r_t)
        navs[rb_next] = nav * (1.0 - C.COST)
    return pd.Series(navs).sort_index()


def main():
    env = C.Env()
    print("加载财务指标缓存...")
    fina = load_fina()
    print("fina 股票数:", fina["ts_code"].nunique())

    print("构建质量分面板...")
    q_map = {}
    for rb in env.rebal:
        if rb < SUB_START:
            continue
        q_map[rb] = quality_map(env, fina, rb)
    # 覆盖率
    cov = [1.0 if (q_map.get(rb) is not None and (q_map[rb] != 0).mean() > 0.2) else 0.0 for rb in q_map]
    print(f"质量分可用调仓月: {sum(cov)}/{len(q_map)}")

    print("重新选股...")
    picks_base = build_picks(env, q_map, use_q=False)
    picks_q = build_picks(env, q_map, use_q=True)
    overlap = []
    for rb in picks_base:
        if rb in picks_q:
            overlap.append(len(set(picks_base[rb]) & set(picks_q[rb])) / C.TOP_N)
    if overlap:
        print(f"月度成分重叠率均值: {np.mean(overlap):.1%}")

    navs = {
        "BASE+VAL": backtest_sub(env, picks_base, use_ma20=False),
        "BASE+VAL+Q": backtest_sub(env, picks_q, use_ma20=False),
        "BASE+VAL+MA20": backtest_sub(env, picks_base, use_ma20=True),
        "BASE+VAL+Q+MA20": backtest_sub(env, picks_q, use_ma20=True),
    }
    rows = [(lb, navs[lb], {}) for lb in navs]
    txt, _ = C.metrics_table(rows)
    report = []
    report.append("=" * 84)
    report.append(f"方向C: 质量因子 QMJ 加入选股 (子区间 {SUB_START[:4]}-07~2026-06)")
    report.append(f"质量分 = zscore(roe) + zscore(毛利率) + zscore(净利同比), ann_date<=rb 最新一期, 缺失补0 (无前视)")
    report.append("")
    report.append(txt)
    report.append("=" * 84)
    out = "\n".join(report)
    with open(os.path.join(C.OUT_DIR, "direction6_quality.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)

    fig, ax = plt.subplots(figsize=(13, 6))
    for lb, nav in navs.items():
        ax.plot(np.arange(len(nav)), nav.values, label=lb, lw=1.6)
    ax.set_title("质量因子 QMJ 增量 (2023-07~2026-06 子区间)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "direction6_quality.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
