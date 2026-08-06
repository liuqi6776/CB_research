# -*- coding: utf-8 -*-
"""阶段5 (P0-3): 集中度约束 — 对 IVW120 目标权重应用组合级上限

每期调仓日、权重计算后按序应用 (迭代收敛):
  1. 单股 ≤ cap_stock (默认 4%)
  2. 行业 ≤ cap_ind   (默认 20%, 复用 engine.cap_industry)
  3. Top5 ≤ cap_top5  (默认 20%: Top5 等比压缩至上限, 权重让渡给其余持仓)
  4. 容量 ≤ cap_amount × 60日均成交额 / (NAV_pre × scale)
     含义: 目标资金 w[c]×NAV×scale 不超过日均成交额的 5% (防止单票冲击成本)
     scale 为账户初始资金规模假设 (回测 NAV 无量纲, 实盘口径需给定)

约束后整体归一化. 不改变标的集合 (剔标的是阶段4 tradability 的职责).
"""
import numpy as np
import pandas as pd

from research.studies.study_008_enhancements.engine import cap_industry


def _cap_top5(w, cap_top5):
    """Top5 合计上限: 折算为单股上限 cap_top5/5 (前5名各自≤该上限 ⇒ 和≤cap_top5),
    与单股 cap 取小者. 绝对上限 → clip 型迭代必然收敛."""
    return min(1.0, cap_top5 / 5.0)


def apply_concentration(w, ind_map=None, cap_stock=0.04, cap_ind=0.20,
                        cap_top5=0.20, amount60=None, nav_pre=1.0,
                        cap_amount=0.05, scale=1e8, iters=10):
    """返回约束后权重 Series (与 w 同 index, 归一化).
    amount60: rb 前 60 日均成交额 Series (元); None 则跳过容量约束.
    scale   : 账户初始资金规模假设 (元), 默认 1 亿.
    顺序: 单股/Top5 折算上限 → 行业 → 容量 (均为绝对上限, 迭代收敛).
    """
    w = w.copy().astype(float)
    cap_eff = cap_stock if cap_stock is not None else None
    if cap_top5 is not None and cap_top5 < 1.0:
        c5 = _cap_top5(w, cap_top5)
        cap_eff = c5 if cap_eff is None else min(float(cap_eff), c5)
    cap_frac = None
    if amount60 is not None and nav_pre > 0:
        denom = nav_pre * scale
        cap_frac = pd.Series(
            {c: (cap_amount * a / denom if np.isfinite(a) else np.inf)
             for c, a in ((c, float(amount60.get(c, np.inf))) for c in w.index)},
            index=w.index)
    for _ in range(iters):
        if cap_eff is not None and cap_eff < 1.0:
            w = w.clip(upper=cap_eff)
        if cap_ind is not None and ind_map is not None:
            w = cap_industry(w, ind_map, cap_ind, iters=5)
        if cap_frac is not None:
            w = w.clip(upper=cap_frac.reindex(w.index).fillna(np.inf))
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            break
    return w


def amount60_at(amount_df, td, rb, lookback=60):
    """rb 前 lookback 个交易日日均成交额 Series (元). td 支持 list / pd.Index"""
    if isinstance(td, pd.Index):
        idx = td.get_loc(rb) if rb in td else len(td) - 1
    else:
        idx = td.index(rb) if rb in td else len(td) - 1
    win = td[max(0, idx - lookback):idx]
    return amount_df.reindex(win).mean()
