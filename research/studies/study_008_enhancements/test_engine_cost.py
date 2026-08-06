# -*- coding: utf-8 -*-
"""阶段2 (P0-1) 单测: 资金权重换手 + 买卖分项费率

验证 asset_turnover_cost 的 4 类关键场景 (不依赖行情数据, 纯函数):
  - 全保留但权重大变: 旧数量口径成本=0, 新资金口径>0
  - 部分换股: 买卖分侧 = ½Σ|Δw| 各自加总
  - 腿切换 (股票↔ETF): 整腿 100% 搬移, ETF 用独立费率
  - 首次建仓 / fail-closed 沿用: 全仓买入 / 零成本
  - 印花税分档: 20230828 前后卖出费率不同
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd

from research.studies.study_008_enhancements.engine import asset_turnover_cost
from research.studies.study_008_enhancements.risk_control_real import (
    COMMISSION, ETF_FEE, stamp_sell, stock_buy_fee, stock_sell_fee,
)


def _w(codes, weights):
    return pd.Series(dict(zip(codes, weights)))


# ---------- 场景1: 全保留但权重大变 (旧数量口径成本=0, 新口径>0) ----------
def test_same_codes_but_weight_shift():
    prev_w = _w(["A", "B"], [0.5, 0.5])
    w = _w(["A", "B"], [0.9, 0.1])
    b, s, be, se, cost = asset_turnover_cost(prev_w, w, False, False, "20230630")
    assert abs(b - 0.4) < 1e-12 and abs(s - 0.4) < 1e-12  # A 增持 0.4, B 减持 0.4
    assert abs(be) < 1e-12 and abs(se) < 1e-12
    assert abs(cost - 0.4 * (stock_buy_fee("20230630") + stock_sell_fee("20230630"))) < 1e-12
    print("[PASS] 全保留权重大变: buy=sell=0.4, cost>0")


# ---------- 场景2: 部分换股 (并集对齐, 买卖分侧) ----------
def test_partial_rebalance():
    prev_w = _w(["A", "B", "C"], [1 / 3, 1 / 3, 1 / 3])
    w = _w(["C", "D", "E"], [1 / 3, 1 / 3, 1 / 3])
    b, s, be, se, cost = asset_turnover_cost(prev_w, w, False, False, "20231231")
    # 卖出 A,B (2/3) + C 的 0 变动; 买入 D,E (2/3)
    assert abs(b - 2 / 3) < 1e-12 and abs(s - 2 / 3) < 1e-12
    assert abs(b - s) < 1e-12  # 全仓模型买卖平衡
    print("[PASS] 部分换股: buy=sell=2/3")


# ---------- 场景3: 腿切换 (股票→ETF / ETF→股票) ----------
def test_leg_switch():
    prev_w = _w(["A", "B"], [0.5, 0.5])
    # 股票→ETF: 全卖股票, 全买 ETF
    b, s, be, se, cost = asset_turnover_cost(prev_w, None, False, True, "20240131")
    assert abs(s - 1.0) < 1e-12 and abs(be - 1.0) < 1e-12 and b == 0 and se == 0
    assert abs(cost - (stock_sell_fee("20240131") + ETF_FEE)) < 1e-12
    # ETF→股票 (引擎中 ETF 月 prev_w 仍存目标股票权重, 旧腿由 prev_etf 决定)
    prev_w = _w(["A", "B"], [0.5, 0.5])  # ETF 月的目标股票权重 (实际持仓 100% ETF)
    b, s, be, se, cost = asset_turnover_cost(prev_w, _w(["A"], [1.0]), True, False, "20240229")
    assert abs(se - 1.0) < 1e-12 and abs(b - 1.0) < 1e-12 and s == 0 and be == 0
    print("[PASS] 腿切换: 整腿 100% 搬移, 分项费率正确")


# ---------- 场景4: 首次建仓 / fail-closed 沿用 ----------
def test_first_position_and_carry():
    # 首次建仓 (股票腿)
    b, s, be, se, cost = asset_turnover_cost(None, _w(["A"], [1.0]), False, False, "20200131")
    assert abs(b - 1.0) < 1e-12 and s == 0 and cost == stock_buy_fee("20200131")
    # fail-closed: w == prev_w → 零成本
    prev_w = _w(["A", "B"], [0.6, 0.4])
    b, s, be, se, cost = asset_turnover_cost(prev_w, prev_w.copy(), False, False, "20210226")
    assert abs(b) < 1e-12 and abs(s) < 1e-12 and abs(cost) < 1e-12
    print("[PASS] 首次建仓 1.0 买入; fail-closed 沿用零成本")


# ---------- 场景5: 印花税分档 (20230828 前后) ----------
def test_stamp_tax_tier():
    assert stamp_sell("20230825") == 0.0010
    assert stamp_sell("20230828") == 0.0005
    assert stock_sell_fee("20230131") == COMMISSION + 0.0010
    assert stock_sell_fee("20250829") == COMMISSION + 0.0005
    # 成本随印花税下降: 2023 后卖出更便宜
    prev_w = _w(["A"], [1.0])
    w = _w(["B"], [1.0])
    _, _, _, _, c1 = asset_turnover_cost(prev_w, w, False, False, "20230131")
    _, _, _, _, c2 = asset_turnover_cost(prev_w, w, False, False, "20240131")
    assert c1 > c2
    print("[PASS] 印花税 2023-08-28 分档: 千1 → 万5")


if __name__ == "__main__":
    test_same_codes_but_weight_shift()
    test_partial_rebalance()
    test_leg_switch()
    test_first_position_and_carry()
    test_stamp_tax_tier()
    print("\n阶段2 成本口径单测 5/5 通过")
