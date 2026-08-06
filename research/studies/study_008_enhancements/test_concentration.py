# -*- coding: utf-8 -*-
"""阶段5 单测: 集中度约束 apply_concentration + amount60_at

覆盖: 单股cap / 行业cap / Top5 cap / 容量cap / 组合场景 / amount60_at.
纯函数测试, 不依赖真实数据. n=60 贴近真实 Top60 (保证 clip 型约束可解:
n×cap>1).
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements.concentration import (
    apply_concentration, amount60_at,
)


def make_w(n=60):
    """60 只股票: 前5只主导 (raw 0.30/0.25/0.20/0.16/0.12), 其余 uniform(0.01,0.05).
    保证: top5 初始 >0.20, max >0.04, 且 n×cap_stock=2.4>1 约束可解."""
    codes = [f"S{i:02d}" for i in range(1, n + 1)]
    rng = np.random.default_rng(1)
    vals = rng.uniform(0.01, 0.05, n)
    vals[:5] = [0.30, 0.25, 0.20, 0.16, 0.12]
    return pd.Series(vals / vals.sum(), index=codes)


def make_ind(w):
    """5 个行业 A..E, 每组 12 只 (行业 cap 0.20 → 5×0.20=1.0 恰好可解).
    主导 5 只都在行业 A, 保证 A 超限被约束."""
    n = len(w)
    grp = n // 5
    return {c: chr(65 + i // grp) for i, c in enumerate(w.index)}


class TestConcentration(unittest.TestCase):

    def test_stock_cap(self):
        w = make_w()
        w2 = apply_concentration(w, cap_stock=0.04, cap_ind=None, cap_top5=None)
        self.assertLessEqual(w2.max(), 0.04 + 1e-6)   # 归一化过冲 < 1e-6
        self.assertAlmostEqual(w2.sum(), 1.0, places=9)

    def test_industry_cap(self):
        w = make_w()
        ind = make_ind(w)
        w2 = apply_concentration(w, ind_map=ind, cap_stock=None, cap_ind=0.20,
                                 cap_top5=None)
        g = w2.groupby(pd.Series({c: ind[c] for c in w2.index})).sum()
        # cap_industry 内部含归一化, 允许 1e-4 级过冲
        self.assertLessEqual(g.max(), 0.20 + 5e-4)
        self.assertAlmostEqual(w2.sum(), 1.0, places=9)

    def test_top5_cap(self):
        w = make_w()
        self.assertGreater(w.nlargest(5).sum(), 0.20)   # 前提: 初始 top5 超标
        w2 = apply_concentration(w, cap_stock=None, cap_ind=None, cap_top5=0.20)
        self.assertLessEqual(w2.nlargest(5).sum(), 0.20 + 1e-6)
        self.assertAlmostEqual(w2.sum(), 1.0, places=9)
        # 权重让渡给其余持仓
        self.assertGreater(w2.iloc[5:].sum(), w.iloc[5:].sum())

    def test_capacity_cap(self):
        w = make_w()
        # 前5只成交额 5e7 (受限), 其余 1e9 (宽松):
        # cap_frac = 0.05*a/(nav_pre*scale) = 0.05*5e7/(2*1e8)=0.0125
        amount60 = pd.Series({c: (5e7 if i < 5 else 1e9)
                              for i, c in enumerate(w.index)})
        w2 = apply_concentration(w, cap_stock=None, cap_ind=None, cap_top5=None,
                                 amount60=amount60, nav_pre=2.0, cap_amount=0.05,
                                 scale=1e8)
        self.assertLessEqual(w2.iloc[:5].max(), 0.0125 + 1e-6)
        self.assertAlmostEqual(w2.sum(), 1.0, places=9)

    def test_all(self):
        w = make_w()
        ind = make_ind(w)
        amount60 = pd.Series({c: (5e7 if i < 5 else 1e9)
                              for i, c in enumerate(w.index)})
        w2 = apply_concentration(w, ind_map=ind, cap_stock=0.04, cap_ind=0.20,
                                 cap_top5=0.20, amount60=amount60, nav_pre=2.0,
                                 cap_amount=0.05, scale=1e8)
        self.assertLessEqual(w2.max(), 0.04 + 1e-6)
        self.assertLessEqual(w2.nlargest(5).sum(), 0.20 + 1e-6)
        g = w2.groupby(pd.Series({c: ind[c] for c in w2.index})).sum()
        self.assertLessEqual(g.max(), 0.20 + 5e-4)
        self.assertLessEqual(w2.iloc[:5].max(), 0.0125 + 1e-5)   # 容量约束取小者
        self.assertAlmostEqual(w2.sum(), 1.0, places=9)

    def test_amount60_at(self):
        n = 70
        td = pd.Index([f"2024{(d//28)+1:02d}{d%28+1:02d}" for d in range(n)])
        amt = pd.DataFrame({"A": np.full(n, 1e7), "B": np.full(n, 2e7)}, index=td)
        rb = td[-1]
        a = amount60_at(amt, td, rb, lookback=60)
        self.assertAlmostEqual(a["A"], 1e7)
        self.assertAlmostEqual(a["B"], 2e7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
