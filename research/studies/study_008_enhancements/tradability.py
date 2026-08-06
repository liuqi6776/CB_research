# -*- coding: utf-8 -*-
"""阶段4 (可交易过滤): 信号名单 → 订单名单

分层原则 (用户验收 P0-2/P0-4, fail-closed):
  - 信号名单 (signal list): 冻结基线因子 TopN (env.picks_map, 不可修改)
  - 订单名单 (order list) : 信号名单剔除 rb 时可观测的不可交易标的
      * ST            : st_history 区间命中 (rb 时已知, 不生成订单)
      * 退市/长期停牌 : rb 前 lookback 日有效成交天数 < min_px_days
      * 极低流动性   : rb 前 lookback 日均成交额 < min_amount (可交易性下限,
                       容量约束按仓位×NAV≤5% 日均成交额在阶段5处理)
  执行时才知道的阻塞 (一字涨跌停 / 当日停牌) 仍在 ledger 执行层处理
  (BLOCKED 顺延), 不在信号端过滤 (避免用未来信息).

订单名单为空 / 过滤后过少 → ledger.run_ledger 整期 fail-closed 沿用上期持仓.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements.engine import load_st_intervals, is_st

_AMOUNT_CACHE = {}


def load_amount_df(env, td):
    """成交额宽表 (列=ts_code, 索引=交易日), 单位统一为元, 进程内缓存.
    tushare 日频 amount 单位=千元, 这里 ×1000 转元 (min_amount 参数按元理解)."""
    key = tuple(env.all_codes)
    if key in _AMOUNT_CACHE:
        return _AMOUNT_CACHE[key]
    stocks, _, _, _, _ = rv.load_panels(td, env.all_codes, None)
    out = pd.DataFrame({c: g.sort_index()["amount"]
                        for c, g in stocks.items() if "amount" in g.columns}) * 1000.0
    _AMOUNT_CACHE[key] = out
    return out


class Tradability:
    """rb 时点可观测的可交易过滤. 用法:
        tf = Tradability(td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                         min_vol=None, pct_df=None)
        order_picks, removed = tf(rb, picks)
    removed: {code: reason}, reason ∈ ST / SUSPEND(退市·长期停牌) / LOW_LIQ(极低流动性)
                                / LOW_VOL(僵尸股·年化波动低于下限)
    min_vol: 可选, rb 前 120 日收益年化波动下限 (P0-3 波动率下限).
      单位口径: pct_df 为百分数 (tushare pct_chg 原始值, 如 1.5 = 1.5%),
      min_vol 同为百分数数值 (12.0 = 12%), 两者同单位直接比较.
    """

    def __init__(self, td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                 st_map=None, min_vol=None, pct_df=None):
        self.td = td
        self.amount_df = amount_df
        self.lookback = lookback
        self.min_amount = float(min_amount)
        self.min_px_days = int(min_px_days)
        self.st_map = st_map if st_map is not None else load_st_intervals()
        self.min_vol = float(min_vol) if min_vol is not None else None
        self.pct_df = pct_df

    def __call__(self, rb, picks):
        if isinstance(self.td, pd.Index):
            idx = self.td.get_loc(rb) if rb in self.td else len(self.td) - 1
        else:  # list
            idx = self.td.index(rb) if rb in self.td else len(self.td) - 1
        win = self.td[max(0, idx - self.lookback):idx]
        win120 = self.td[max(0, idx - 120):idx] if self.min_vol is not None else None
        removed = {}
        keep = []
        for c in picks:
            if is_st(self.st_map, c, rb):
                removed[c] = "ST"
                continue
            amt = self.amount_df[c].reindex(win) if c in self.amount_df.columns \
                else pd.Series(dtype=float)
            amt_valid = amt.dropna()
            if len(amt_valid) < self.min_px_days:
                removed[c] = "SUSPEND"   # 退市/长期停牌: 窗口内有效成交天数不足
                continue
            if amt_valid.mean() < self.min_amount:
                removed[c] = "LOW_LIQ"   # 极低流动性: 不可交易性下限
                continue
            if self.min_vol is not None and self.pct_df is not None:
                r = self.pct_df[c].reindex(win120).dropna()
                if len(r) >= 30 and float(r.std() * np.sqrt(252)) < self.min_vol:
                    removed[c] = "LOW_VOL"   # 僵尸股: 年化波动低于下限
                    continue
            keep.append(c)
        return keep, removed
