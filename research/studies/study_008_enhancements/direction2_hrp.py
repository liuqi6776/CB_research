# -*- coding: utf-8 -*-
"""方向2: 层次风险平价 (HRP) 替代 Top50 等权
- 权重: 调仓日取截至 rb 的过去 120 日个股日收益 (T-1 已知, 无前视), LedoitWolf 协方差 + 层次聚类 + 递归二分
- 对比: 等权 vs HRP (无 MA20); 等权+MA20 vs HRP+MA20 (全区间, RS12/成本/基准一致)
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

from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from research.studies.study_008_enhancements import common as C

WINDOW = 120
MA20_DEEP = 0.98


def _hrp_weights(returns):
    """returns: DataFrame(日收益, 行=交易日, 列=个股). 返回个股权重 Series"""
    r = returns.dropna(how="all")
    if len(r) < 20 or len(r.columns) < 2:
        return pd.Series(1.0 / len(r.columns), index=r.columns)
    # 协方差 shrinkage
    try:
        cov = LedoitWolf().fit(r).covariance_
    except Exception:
        cov = r.cov().values + np.eye(len(r.columns)) * 1e-6
    corr = pd.DataFrame(r.corr().values, index=r.columns, columns=r.columns).fillna(0.0)
    dist = pd.DataFrame(np.sqrt(0.5 * (1.0 - corr)), index=corr.index, columns=corr.columns)
    cov_pd = pd.DataFrame(cov, index=r.columns, columns=r.columns)
    # 层次聚类
    try:
        links = linkage(squareform(dist.values, checks=False), method="single")
    except Exception:
        links = linkage(dist.values[np.triu_indices_from(dist.values, k=1)], method="single")
    sort_idx = _get_cluster_sort(links, len(r.columns))
    sorted_codes = r.columns[sort_idx]

    def _recursive_bisect(codes):
        if len(codes) == 1:
            return {codes[0]: 1.0}
        # 按聚类顺序二分
        mid = len(codes) // 2
        left, right = codes[:mid], codes[mid:]
        wl = _inverse_vol(cov_pd, left)
        wr = _inverse_vol(cov_pd, right)
        total = wl + wr
        if total <= 0:
            return {c: 1.0 / len(codes) for c in codes}
        lw = wl / total
        d_l = _recursive_bisect(left)
        d_r = _recursive_bisect(right)
        out = {}
        for c, w in d_l.items():
            out[c] = lw * w
        for c, w in d_r.items():
            out[c] = (1 - lw) * w
        return out

    wmap = _recursive_bisect(sorted_codes)
    w = pd.Series(wmap).reindex(r.columns).fillna(0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    return w


def _get_cluster_sort(links, n):
    """由 linkage 矩阵恢复叶子排序 (leaves_list)"""
    return list(leaves_list(links))


def _inverse_vol(cov, codes):
    sub = cov.loc[codes, codes]
    vols = np.sqrt(np.diag(sub))
    iv = 1.0 / (vols + 1e-9)
    return iv.sum()


def run_hrp(env, use_hrp, use_ma20):
    navs = {}
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        nav = navs.get(rb, 1.0)
        if picks is None:
            navs[rb_next] = nav
            continue
        if use_hrp:
            # 截至 rb 的过去 WINDOW 日收益 (T-1 可得)
            hi = env.trade_dates.index(rb)
            win = env.trade_dates[max(0, hi - WINDOW):hi]
            rets = env.pct_df.reindex(columns=picks).reindex(win)
            w = _hrp_weights(rets)
        else:
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
    navs = {
        "等权": run_hrp(env, use_hrp=False, use_ma20=False),
        "HRP": run_hrp(env, use_hrp=True, use_ma20=False),
        "等权+MA20": run_hrp(env, use_hrp=False, use_ma20=True),
        "HRP+MA20": run_hrp(env, use_hrp=True, use_ma20=True),
    }
    rows = [(lb, nav, {}) for lb, nav in navs.items()]
    txt, _ = C.metrics_table(rows)
    report = []
    report.append("=" * 80)
    report.append("方向2: 层次风险平价 (HRP) 替代 Top50 等权")
    report.append(f"权重: 调仓日取截至 rb 过去 {WINDOW} 日收益, LedoitWolf 协方差 + 单连接聚类 + 递归二分; T-1 可得, 无前视")
    report.append("月内权重固定 (与等权一致), 月度调仓 20bps, RS12 弱段持 512100")
    report.append("")
    report.append(txt)
    report.append("=" * 80)
    out = "\n".join(report)
    with open(os.path.join(C.OUT_DIR, "direction2_hrp.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)

    fig, ax = plt.subplots(figsize=(13, 6))
    for lb, nav in navs.items():
        ax.plot(np.arange(len(nav)), nav.values, label=lb, lw=1.6)
    ax.set_title("HRP vs 等权 (BASE+VAL, 2020-2026)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "direction2_hrp.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
