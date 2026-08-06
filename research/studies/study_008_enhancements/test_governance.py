# -*- coding: utf-8 -*-
"""阶段7 账户治理层单测: 峰值回撤分级状态机 + 实盘日报字段"""
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(__file__ and r"c:\Users\liuqi\quant_system_v2")
from research.studies.study_008_enhancements.governance import Governance, daily_report


class TestGovernance(unittest.TestCase):

    def test_levels(self):
        g = Governance()
        nav = pd.Series(np.linspace(1.0, 1.5, 50))       # 上行, 无回撤
        for i, n in nav.items():
            lv, dd = g.on_nav(i, n)
            self.assertEqual(lv, 0)
        # 从 1.5 峰值回撤: 边界按 dd=1-peak/nav:
        #   nav=1.32 → dd=13.6%  → L0; nav=1.30 → dd=15.4% → L1;
        #   nav=1.24 → dd=21.0%  → L2; nav=1.19 → dd=26.1% → L3
        cases = [(1.32, 0), (1.30, 1), (1.24, 2), (1.19, 3)]
        for nav_val, expect in cases:
            lv, dd = g.on_nav(100 + nav_val * 10, nav_val)
            self.assertEqual(lv, expect, f"nav={nav_val} expect L{expect} got L{lv}")
        self.assertAlmostEqual(g.peak, 1.5)

    def test_event_ratchet(self):
        g = Governance()
        # 触发 L3 (0.97: dd=34%) 后回撤缓解 (1.1: dd=18.2% → L1) 降级, 事件历史保留
        for i, n in [(0, 1.0), (1, 1.3), (2, 0.97), (3, 1.1)]:
            g.on_nav(i, n)
        self.assertEqual(g.state, 1)
        evs = [e for e in g.events if e[1] == 3]
        self.assertTrue(evs)
        self.assertAlmostEqual(evs[0][2], 1.3 / 0.97 - 1.0)

    def test_daily_report_fields(self):
        g = Governance()
        navs = pd.Series([1.0, 1.2, 1.1, 1.15], index=pd.Index(["20260101", "20260102",
                                                                "20260103", "20260104"]))
        for t, n in navs.items():
            g.on_nav(t, n)
        stats = dict(final_cash=0.2, final_etf=10.0, n_units=5, n_pending=1,
                     n_missing=1, n_leg_block=0, n_buy_block=2, n_sell_block=1,
                     n_st_block=0, n_susp_block=1, fees=3.14, avg_te=0.004,
                     blocked=None)
        rep = daily_report(navs, stats, g, "20260104")
        for k in ("date", "nav", "peak_nav", "drawdown_pct", "gov_level",
                  "gov_action", "cash", "etf_units", "n_stocks",
                  "n_pending_orders", "fail_closed_days", "blocked_buy",
                  "blocked_sell", "blocked_st", "blocked_susp", "fees_total",
                  "avg_te_pct", "alert"):
            self.assertIn(k, rep, f"日报缺字段 {k}")
        self.assertEqual(rep["date"], "20260104")
        self.assertEqual(rep["fail_closed_days"], 1)
        self.assertEqual(rep["alert"], 1)   # n_pending>0 → 并发阻断告警
        # 回撤 1.2->1.15 = 4.2% → L0
        self.assertEqual(rep["gov_level"], 0)


if __name__ == "__main__":
    unittest.main()
