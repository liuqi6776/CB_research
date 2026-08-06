# -*- coding: utf-8 -*-
"""阶段3 (P0-2) 单测: 双账本持仓簿 + 订单状态机 (不依赖行情, 合成价格)

覆盖用户验收清单 P0-2 关键规则:
  - 账本恒等: NAV = cash + Σ份额×价格 (调仓后)
  - 买不进 → 现金保留 (不重新分配), pending 订单 BLOCKED 顺延
  - 卖不出 → 旧仓继续承担收益风险 (不现金化)
  - 目标退出后 pending 订单 DROPPED
  - tracking_error: 目标 vs 实际 ½Σ|Δw| (含现金腿)
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from research.studies.study_008_enhancements.ledger import (
    PortfolioBook, REASON_ST, REASON_UP, REASON_DN, REASON_SUSP,
)
from research.studies.study_008_enhancements.risk_control_real import (
    COMMISSION, stock_buy_fee, stock_sell_fee,
)

T0 = "20240628"


def mk_open_close(codes, price_map):
    """单日 open/close 宽表: 一行 t0"""
    idx = pd.Index([T0], name="date")
    open_df = pd.DataFrame({c: [price_map[c][0]] for c in codes}, index=idx)
    close_df = pd.DataFrame({c: [price_map[c][1]] for c in codes}, index=idx)
    return open_df, close_df


def mk_etf(open_px=1.0):
    return pd.Series({T0: open_px})


# ---------- 场景1: 正常建仓, 账本恒等 + NAV=1 ----------
def test_initial_rebalance_identity():
    codes = ["A", "B", "C"]
    price_map = {c: (10.0, 10.5) for c in codes}
    open_df, close_df = mk_open_close(codes, price_map)
    book = PortfolioBook()
    w = pd.Series([0.4, 0.35, 0.25], index=codes)
    te, cost, n_block = book.rebalance("202406", T0, w, False, open_df, close_df, mk_etf())
    # 份额 = 权重/开盘价
    assert abs(book.units["A"] - 0.4 / 10.0) < 1e-12
    # 账本恒等 (用开盘价): cash + Σ份额×价 = 1
    val = sum(book.units[c] * 10.0 for c in codes)
    assert abs(book.cash + val - 1.0 + cost) < 1e-9  # cost 已扣现金
    # 成本 = 买入 1.0 × 佣金
    assert abs(cost - stock_buy_fee("202406")) < 1e-12
    assert te < 1e-2 and n_block == 0  # 无阻塞: TE 仅含成本拖累残差 (~2.5bp)
    print("[PASS] 正常建仓: 账本恒等, 成本=全仓买入佣金, TE≈0")


# ---------- 场景2: 买不进 (一字涨停) → 现金保留 + pending BLOCKED ----------
def test_buy_block_keeps_cash():
    codes = ["A", "B", "C"]
    price_map = {c: (10.0, 10.5) for c in codes}
    open_df, close_df = mk_open_close(codes, price_map)
    book = PortfolioBook()
    w = pd.Series([0.5, 0.3, 0.2], index=codes)
    one_up = {("B", T0)}  # B 一字涨停不可买
    te, cost, n_block = book.rebalance("202406", T0, w, False, open_df, close_df,
                                       mk_etf(), one_up=one_up)
    assert "B" not in book.units                      # 未买入
    assert "A" in book.units and "C" in book.units    # 其余正常买入
    assert n_block == 1 and book.n_buy_block == 1
    assert "B" in book.pending and book.pending["B"].status == "BLOCKED"
    # 现金保留 = 未买成的 0.3 (不被重新分配); 恒等式 cash+val+cost=1
    val = book.units["A"] * 10.0 + book.units["C"] * 10.0
    assert abs(book.cash + val + cost - 1.0) < 1e-9
    assert abs(book.cash - (0.3 - cost)) < 1e-9
    assert te > 0.15  # B 缺失 0.3 → ½×0.3 + cash 贡献
    print(f"[PASS] 买不进现金保留: cash={book.cash:.3f}, pending=BLOCKED, TE={te:.3f}")


# ---------- 场景3: 卖不出 (一字跌停) → 旧仓继续持有, 不现金化 ----------
def test_sell_block_keeps_position():
    codes = ["A", "B", "C"]
    price_map = {c: (10.0, 10.5) for c in codes}
    open_df, close_df = mk_open_close(codes, price_map)
    book = PortfolioBook()
    w0 = pd.Series([1 / 3, 1 / 3, 1 / 3], index=codes)
    book.rebalance("202405", T0, w0, False, open_df, close_df, mk_etf())
    # 次月 A 退出目标, 但 A 一字跌停卖不出
    w1 = pd.Series([0.0, 0.5, 0.5], index=["A", "B", "C"])
    one_dn = {("A", T0)}
    book.rebalance("202406", T0, w1, False, open_df, close_df, mk_etf(), one_dn=one_dn)
    assert "A" in book.units           # 旧仓保留
    assert book.n_sell_block == 1
    # A 的市值没有被现金化 (仍在 units 中承担收益风险)
    assert book.units["A"] * 10.0 > 0
    print(f"[PASS] 卖不出旧仓保留: units[A]={book.units['A']:.5f}, sell_block=1")


# ---------- 场景4: 目标退出 → pending DROPPED ----------
def test_pending_dropped_when_exit():
    codes = ["A", "B", "C"]
    price_map = {c: (10.0, 10.5) for c in codes}
    open_df, close_df = mk_open_close(codes, price_map)
    book = PortfolioBook()
    w0 = pd.Series([0.5, 0.5], index=["A", "B"])
    one_up = {("B", T0)}
    book.rebalance("202405", T0, w0, False, open_df, close_df, mk_etf(), one_up=one_up)
    assert "B" in book.pending
    # 次月 B 退出目标 → DROPPED
    w1 = pd.Series([1.0], index=["A"])
    book.rebalance("202406", T0, w1, False, open_df, close_df, mk_etf())
    assert "B" not in book.pending
    assert book.pending.get("B") is None
    print("[PASS] 目标退出: pending 订单 DROPPED 作废")


# ---------- 场景5: 腿切换 股票→ETF→股票 ----------
def test_leg_switch_roundtrip():
    codes = ["A", "B"]
    price_map = {c: (10.0, 10.5) for c in codes}
    open_df, close_df = mk_open_close(codes, price_map)
    book = PortfolioBook()
    w0 = pd.Series([0.5, 0.5], index=codes)
    book.rebalance("202405", T0, w0, False, open_df, close_df, mk_etf())
    # → ETF
    book.rebalance("202406", T0, pd.Series(dtype=float), True, open_df, close_df, mk_etf(2.0))
    assert book.etf_units > 0 and len(book.units) == 0
    assert abs(book.etf_units * 2.0 + book.cash - 1.0 + book.fees) < 1e-9
    # → 回股票
    book.rebalance("202407", T0, w0, False, open_df, close_df, mk_etf(2.0))
    assert book.etf_units == 0 and set(book.units) == set(codes)
    print("[PASS] 腿切换往返: 账本连续, ETF 份额正确")


if __name__ == "__main__":
    test_initial_rebalance_identity()
    test_buy_block_keeps_cash()
    test_sell_block_keeps_position()
    test_pending_dropped_when_exit()
    test_leg_switch_roundtrip()
    print("\n阶段3 双账本单测 5/5 通过")
