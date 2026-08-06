# -*- coding: utf-8 -*-
"""阶段4 单测: Tradability 可交易过滤 (信号名单 → 订单名单)

覆盖: ST 剔除 / 长期停牌退市剔除 / 极低流动性剔除 / 正常保留 / 混合场景
不依赖真实数据: 用构造的 td + amount_df + st_map.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements.tradability import Tradability


def make_env(n=70):
    td = pd.Index([f"2024{(d//28)+1:02d}{d%28+1:02d}" for d in range(n)], name="td")
    codes = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(0)
    amt = pd.DataFrame(
        {c: rng.uniform(1e7, 5e7, n) for c in codes}, index=td)
    # C: 极低流动性 (日均 ~5e5 < 3e6)
    amt["C"] = 5e5
    # D: 长期停牌/退市 (rb 前窗口内仅 2 个有效成交日)
    amt["D"] = np.nan
    amt.iloc[:2, amt.columns.get_loc("D")] = 2e7
    return td, amt


class TestTradability(unittest.TestCase):

    def setUp(self):
        self.td, self.amt = make_env()
        self.rb = self.td[-1]   # rb 取最后一日, 窗口 = 前 60 个交易日

    def test_normal_kept(self):
        tf = Tradability(self.td, self.amt)
        keep, removed = tf(self.rb, ["A", "B"])
        self.assertEqual(sorted(keep), ["A", "B"])
        self.assertEqual(removed, {})

    def test_st_removed(self):
        tf = Tradability(self.td, self.amt,
                         st_map={"B": [("20240101", "20241231")]})
        keep, removed = tf(self.rb, ["A", "B", "E"])
        self.assertEqual(keep, ["A", "E"])
        self.assertEqual(removed["B"], "ST")

    def test_low_liq_removed(self):
        tf = Tradability(self.td, self.amt)
        keep, removed = tf(self.rb, ["A", "C"])
        self.assertEqual(keep, ["A"])
        self.assertEqual(removed["C"], "LOW_LIQ")

    def test_suspend_removed(self):
        tf = Tradability(self.td, self.amt)
        keep, removed = tf(self.rb, ["A", "D"])
        self.assertEqual(keep, ["A"])
        self.assertEqual(removed["D"], "SUSPEND")

    def test_mixed(self):
        tf = Tradability(self.td, self.amt,
                         st_map={"E": [("20240101", "20241231")]})
        keep, removed = tf(self.rb, ["A", "B", "C", "D", "E"])
        self.assertEqual(keep, ["A", "B"])
        self.assertEqual(removed, {"C": "LOW_LIQ", "D": "SUSPEND", "E": "ST"})

    def test_unknown_code_removed(self):
        """不在成交额面板中的标的 → 视为无行情剔除"""
        tf = Tradability(self.td, self.amt)
        keep, removed = tf(self.rb, ["A", "ZZZ"])
        self.assertEqual(keep, ["A"])
        self.assertEqual(removed["ZZZ"], "SUSPEND")

    def test_low_vol_removed(self):
        """僵尸股过滤 (P0-3 波动率下限): 年化波动 < min_vol → LOW_VOL"""
        n = 140
        td = pd.Index([f"2024{(d//28)+1:02d}{d%28+1:02d}" for d in range(n)], name="td")
        rng = np.random.default_rng(3)
        # A: 正常波动 (~25% 年化), B: 僵尸股 (~5% 年化) — 百分数单位 (pct_chg 口径)
        pct = pd.DataFrame({
            "A": rng.normal(0, 1.6, n),
            "B": rng.normal(0, 0.3, n),
        }, index=td)
        amt = pd.DataFrame({c: 2e7 for c in ["A", "B"]}, index=td)
        rb = td[-1]
        tf = Tradability(td, amt, min_vol=12.0, pct_df=pct)
        keep, removed = tf(rb, ["A", "B"])
        self.assertEqual(keep, ["A"])
        self.assertEqual(removed["B"], "LOW_VOL")
        # 未启用 min_vol → 不过滤
        tf0 = Tradability(td, amt)
        keep0, _ = tf0(rb, ["A", "B"])
        self.assertEqual(sorted(keep0), ["A", "B"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
