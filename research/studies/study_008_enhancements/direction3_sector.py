# -*- coding: utf-8 -*-
"""方向3: 行业轮动过滤 — 仅在强势行业成分股内做因子选股
- 行业映射: tushare stock_basic 东财行业 (静态近似, 研究用)
- 行业动量: 当期成分股按行业等权合成日收益, 过去 63 交易日累计收益 (T-1 已知, 无前视)
- 选股: 动量前 N 行业成分股 ∩ 中证1000 成分 → BASE+VAL 因子 Top50
- 对比: 全池选股 vs 强势行业池(N=3/5/8), 及叠加 MA20
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

MOM_WINDOW = 63
MA20_DEEP = 0.98


def build_industry_picks(env, top_n):
    """每月: 强势行业(动量前 top_n)成分股内选 Top50 (BASE+VAL)"""
    ind_map = C.load_industry_map()
    picks_map = {}
    for rb in env.rebal:
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        # 当期成分股内的行业收益合成 (过去 MOM_WINDOW 交易日)
        hi = env.trade_dates.index(rb)
        win = env.trade_dates[max(0, hi - MOM_WINDOW):hi]
        rets = env.pct_df.reindex(win)
        ind_ret = {}
        for code in members:
            ind = ind_map.get(code)
            if ind is None:
                continue
            s = rets.get(code)
            if s is None:
                continue
            v = s.dropna()
            if len(v) == 0:
                continue
            ind_ret.setdefault(ind, []).append(v)
        if not ind_ret:
            continue
        ind_mom = {ind: pd.concat(vs).groupby(level=0).mean().sum()
                   for ind, vs in ind_ret.items()}
        top_inds = sorted(ind_mom, key=ind_mom.get, reverse=True)[:top_n]
        pool = {c for c in members if ind_map.get(c) in top_inds}
        # 因子选股 (同 common, 池受限)
        fvals = {}
        for code in pool:
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
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < 10:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = sf.BASE_COLS + ["VAL"]
        has = zdf[cols].dropna()
        if len(has) < 10:
            continue
        n = min(C.TOP_N, len(has))
        picks_map[rb] = has.mean(axis=1).nlargest(n).index.tolist()
    return picks_map


def run(env, picks_map, use_ma20):
    navs = {}
    for rb, rb_next, hold, picks0, comb0, e_ret, rs12_on in env.month_segments():
        nav = navs.get(rb, 1.0)
        picks = picks_map.get(rb)
        if picks is None:
            navs[rb_next] = nav
            continue
        comb = env.pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
        cr = comb.mean(axis=1)
        for t in hold:
            r_t = e_ret.loc[t]
            if rs12_on:
                w = 1.0
                if use_ma20:
                    c = env.idx_close_1.get(t, np.nan)
                    m = env.ma20_1.get(t, np.nan)
                    if np.isfinite(c) and np.isfinite(m):
                        w = 1.0 if c >= m else (0.5 if c >= MA20_DEEP * m else 0.0)
                r_t = w * cr.loc[t]
            nav *= (1.0 + r_t)
        navs[rb_next] = nav * (1.0 - C.COST)
    return pd.Series(navs).sort_index()


def main():
    env = C.Env()
    p3 = build_industry_picks(env, 3)
    p5 = build_industry_picks(env, 5)
    p8 = build_industry_picks(env, 8)

    navs = {
        "全池(对照)": run(env, env.picks_map, use_ma20=False),
        "行业Top3": run(env, p3, use_ma20=False),
        "行业Top5": run(env, p5, use_ma20=False),
        "行业Top8": run(env, p8, use_ma20=False),
        "全池+MA20": run(env, env.picks_map, use_ma20=True),
        "行业Top5+MA20": run(env, p5, use_ma20=True),
    }
    rows = [(lb, nav, {}) for lb, nav in navs.items()]
    txt, _ = C.metrics_table(rows)

    # 覆盖率: 每月强势行业池内因子选出的股票数
    cover = []
    for rb in env.rebal:
        c = len(p5.get(rb, []))
        if c > 0:
            cover.append(c)
    report = []
    report.append("=" * 80)
    report.append("方向3: 行业轮动过滤 — 强势行业池内因子选股")
    report.append("行业动量: 当期中证1000成分股按东财行业等权合成日收益, 过去63交易日累计(无前视)")
    report.append("选股: 动量前N行业成分股 ∩ 中证1000 → BASE+VAL Top50; 全区间 2020-2026, 20bps")
    report.append("")
    report.append(txt)
    report.append("")
    report.append(f"行业Top5 池内选出股票数均值: {np.mean(cover):.1f} (Top50 饱和度 {np.mean(cover)/C.TOP_N*100:.0f}%)")
    report.append("=" * 80)
    out = "\n".join(report)
    with open(os.path.join(C.OUT_DIR, "direction3_sector.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)

    fig, ax = plt.subplots(figsize=(13, 6))
    for lb, nav in navs.items():
        ax.plot(np.arange(len(nav)), nav.values, label=lb, lw=1.6)
    ax.set_title("行业轮动过滤 vs 全池 (BASE+VAL, 2020-2026)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "direction3_sector.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
