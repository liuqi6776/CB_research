# -*- coding: utf-8 -*-
"""新因子准入检查工具集 (2026-07-17, 沉淀自新闻因子 P1-P5 证伪案例)

用法:
    from utils.factor_checks import run_factor_admission
    verdict = run_factor_admission(df, factor="rel_sentiment", ret="ret_2d", date="date")

设计原则: 因子有举证责任 —— 它必须在时间稳定性和样本外上证明自己,
而不是我们证明它无效。统计显著 ≠ 可交易。

依赖: 仅 pandas/numpy (环境无 scipy)。
"""
import numpy as np
import pandas as pd
from math import erf, sqrt

# ---------------------------------------------------------------- 基础统计

def pval_norm(t):
    """大样本正态近似双侧 p 值 (无 scipy 环境)"""
    if t is None or np.isnan(t):
        return np.nan
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def sig_stars(p):
    if np.isnan(p):
        return ""
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))


def welch_t(a, b):
    """Welch 双样本 t 检验, 返回 (t, p)"""
    a, b = pd.Series(a).dropna().values, pd.Series(b).dropna().values
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    t = (a.mean() - b.mean()) / sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return t, pval_norm(t)


def one_samp_t(a, mu=0.0):
    a = pd.Series(a).dropna().values
    if len(a) < 3:
        return np.nan, np.nan
    t = (a.mean() - mu) / (a.std(ddof=1) / sqrt(len(a)))
    return t, pval_norm(t)


def winsorize(s, q=(0.01, 0.99)):
    lo, hi = s.quantile(list(q))
    return s.clip(lo, hi)


# ---------------------------------------------------------------- 关卡1: 样本内显著性

def ic_daily(df, factor, ret, date="date", min_n=5):
    """每日截面 Spearman IC (rank 后 Pearson, 免 scipy)。

    返回 dict(ic, t, p, n_days, ic_series)
    注意: df 的 ret 建议先 winsorize, 极端值会污染 IC 与分层。
    """
    ics, dates = [], []
    for d, g in df.groupby(date):
        if len(g) < min_n or g[factor].nunique() < 3 or g[ret].nunique() < 3:
            continue
        ics.append(g[factor].rank().corr(g[ret].rank()))
        dates.append(d)
    s = pd.Series(ics, index=pd.Index(dates, name=date)).dropna()
    if len(s) < 5:
        return {"ic": np.nan, "t": np.nan, "p": np.nan, "n_days": len(s), "ic_series": s}
    t = s.mean() / (s.std(ddof=1) / sqrt(len(s)))
    return {"ic": s.mean(), "t": t, "p": pval_norm(t), "n_days": len(s), "ic_series": s}


def quintile_spread(df, factor, ret, n_q=5):
    """因子分层收益 (pooled)。返回 (各层均值Series, 多空 Q1-Q5)"""
    x = df[[factor, ret]].dropna().copy()
    x["q"] = pd.qcut(x[factor].rank(method="first"), n_q,
                     labels=[f"Q{i+1}" for i in range(n_q)])
    piv = x.groupby("q", observed=True)[ret].mean()
    return piv, piv.iloc[0] - piv.iloc[-1]


# ---------------------------------------------------------------- 关卡2: 时间稳定性

def split_half_stability(df, factor, ret, date="date", min_n=5):
    """按日期中位数分前后两半, 各算 IC。
    通过标准: 两半同号且至少各 |t|>1.65 (p<0.1); 一半不显著 = 红旗 (P2 案例)。
    """
    med = df[date].median()
    out = {}
    for label, wd in [("前半", df[df[date] <= med]), ("后半", df[df[date] > med])]:
        r = ic_daily(wd, factor, ret, date, min_n)
        out[label] = r
    return out


def rolling_ic(df, factor, ret, date="date", window=20, min_n=5):
    """滚动窗口 IC 均值序列 (看 regime 依赖)"""
    r = ic_daily(df, factor, ret, date, min_n)
    return r["ic_series"].rolling(window, min_periods=max(5, window // 2)).mean()


# ---------------------------------------------------------------- 关卡0: 前视检查

def post_entry_return(df, entry_col, exit_col):
    """从观察完成点起算的持有收益: (1+exit)/(1+entry)-1

    规则(血泪教训): 任何"观察到形态后再入场"的信号, 收益一律从观察完成点起算。
    例: 重大负面后观察 T+1~T+3 是否不跌, 入场点=T+3收盘,
        r_3to5 = (1+ret_5d)/(1+ret_3d)-1  (P3 案例: 87% 收益在选择窗口内)
    """
    return ((1 + df[exit_col] / 100) / (1 + df[entry_col] / 100) - 1) * 100


# ---------------------------------------------------------------- 总流程

def run_factor_admission(df, factor, ret, date="date", oos_df=None,
                         min_n=5, verbose=True):
    """新因子准入五道关卡 (打印并返回判定 dict)。

    df:     样本内面板 (必须含 date, factor, ret 列)
    oos_df: 样本外面板 (同结构; 无则关卡3标记为"未执行=不通过")

    通过标准 (全部满足才准入):
      关卡0 收益起算点审查     —— 人工确认, 见 FACTOR_CHECKLIST.md
      关卡1 样本内: |t|>=2 且分层单调
      关卡2 分半: 两半 IC 同号且各 p<0.1
      关卡3 样本外: IC 同号且 |t|>=1.65
      关卡4 可交易性: 人工确认成本/容量/相关性
    """
    v = {}
    d = df.copy()
    d[ret] = winsorize(d[ret])

    full = ic_daily(d, factor, ret, date, min_n)
    piv, ls = quintile_spread(d, factor, ret)
    half = split_half_stability(d, factor, ret, date, min_n)

    g1 = (not np.isnan(full["t"])) and abs(full["t"]) >= 2
    h1, h2 = half["前半"], half["后半"]
    g2 = (not np.isnan(h1["p"])) and (not np.isnan(h2["p"])) \
        and np.sign(h1["ic"]) == np.sign(h2["ic"]) \
        and h1["p"] < 0.1 and h2["p"] < 0.1
    v["gate1_in_sample"] = g1
    v["gate2_split_half"] = g2

    if oos_df is not None and len(oos_df):
        o = oos_df.copy()
        o[ret] = winsorize(o[ret])
        oos = ic_daily(o, factor, ret, date, min_n)
        g3 = (not np.isnan(oos["t"])) and np.sign(oos["ic"]) == np.sign(full["ic"]) \
            and abs(oos["t"]) >= 1.65
        v["gate3_oos"] = g3
    else:
        oos = None
        v["gate3_oos"] = False  # 未执行样本外 = 不通过 (举证责任在因子)

    if verbose:
        print("=" * 60)
        print(f"因子准入检查: {factor} vs {ret}")
        print("-" * 60)
        print(f"[关卡1] 样本内 IC={full['ic']:+.4f}, t={full['t']:+.2f}, "
              f"p={full['p']:.3f} {sig_stars(full['p'])}, n_days={full['n_days']} -> "
              f"{'✅' if g1 else '❌'}")
        print(f"        五分位: " + ", ".join(f"{k}={x:+.3f}" for k, x in piv.items())
              + f"  多空={ls:+.3f}")
        print(f"[关卡2] 分半: 前半 IC={h1['ic']:+.4f}(p={h1['p']:.3f}) / "
              f"后半 IC={h2['ic']:+.4f}(p={h2['p']:.3f}) -> {'✅' if g2 else '❌'}")
        if oos:
            print(f"[关卡3] 样本外 IC={oos['ic']:+.4f}, t={oos['t']:+.2f}, "
                  f"p={oos['p']:.3f} {sig_stars(oos['p'])} -> {'✅' if v['gate3_oos'] else '❌'}")
        else:
            print("[关卡3] 样本外未执行 -> ❌ (因子有举证责任)")
        passed = all(v.values())
        print("-" * 60)
        print(f"机器可判关卡: {'✅ 全过' if passed else '❌ 未全过'}; "
              f"关卡0(前视)与关卡4(可交易性)需人工确认")
    return v
