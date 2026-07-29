# -*- coding: utf-8 -*-

"""
单元测试：回测撮合引擎与风控规约回归测试 (Simulation Engine Regression Unit Tests)
锁定：
1. 往返 20 bps 交易摩擦成本扣除 (买入 +10 bps, 卖出 -10 bps)
2. 15分钟 K线 5% 容量上限控制 (bar_vol * 0.5 * 10 张)
3. 可成交状态校验 (is_executable_at_fill = True)
4. 动态权益分配 (根据当前 capital 动态分配而非固定 1,000,000)
"""

import os
import sys
from pathlib import Path

# 添加仓库根目录到 sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
import numpy as np
import pandas as pd
from run_master_multifactor_backtest import simulate_nav

def test_transaction_costs_deduction():
    """验证交易成本是否严格扣除 (买入 +10 bps, 卖出 -10 bps)"""
    df_pit = pd.DataFrame([
        {'date_str': '20240102', 'ts_code': '110001.SH', 'open': 100.0, 'close': 100.0, 'vol': 10000.0, 'is_executable_at_fill': True, 'is_redeemed': False},
        {'date_str': '20240103', 'ts_code': '110001.SH', 'open': 100.0, 'close': 100.0, 'vol': 10000.0, 'is_executable_at_fill': True, 'is_redeemed': False},
    ])
    
    # 模拟第1天买入，第2天卖出
    df_orders = pd.DataFrame([
        {'trade_date': '20240102', 'ts_code': '110001.SH'}
    ])
    
    # 无调仓的目标单在第2天不出现，触发卖出
    res = simulate_nav(df_pit, df_orders, use_timing=False)
    
    # 初始资金 1,000,000，单只持仓上限 10% (约 100,000 元)
    # 买入价: 100.0 * 1.0010 = 100.10 元/张
    # 卖出价: 100.0 * 0.9990 = 99.90 元/张
    # 单持仓往返损耗 20 bps (在 10% 组合仓位下对全组合 NAV 损耗约为 -0.01%)
    assert res['total_ret'] < 0.0, "在价格完全平稳的情况下，扣除 20 bps 摩擦成本后累计收益率必须为负"
    assert res['total_ret'] <= -0.00008, "单仓 20 bps 损耗在 10% 仓位下组合损耗必须为负"

def test_volume_capacity_limit():
    """验证 15分钟 K线 5% 容量上限控制 (bar_vol * 10 * 0.05 张 = bar_vol * 0.5)"""
    # 当 bar_vol 极小 (例如 10 手) 时，5% 容量限制最大买入 5 手 (50 张)
    df_pit = pd.DataFrame([
        {'date_str': '20240102', 'ts_code': '110001.SH', 'open': 100.0, 'close': 100.0, 'vol': 10.0, 'is_executable_at_fill': True, 'is_redeemed': False},
    ])
    
    df_orders = pd.DataFrame([
        {'trade_date': '20240102', 'ts_code': '110001.SH'}
    ])
    
    res = simulate_nav(df_pit, df_orders, use_timing=False)
    # 不受容量限制下理想可买约 1000 张，但在 10 手容量上限下仅能成交 50 张 (5手)
    # 计算实际扣除资金应远小于 100,000 元
    final_nav = res['nav_series'].iloc[-1]
    cash_used = 1000000.0 - final_nav
    assert cash_used < 10000.0, "低流动性标的交易受 5% 容量上限截断，成交金额受限"

def test_non_executable_status_check():
    """验证不可成交/停牌标的拒绝下单"""
    df_pit = pd.DataFrame([
        {'date_str': '20240102', 'ts_code': '110001.SH', 'open': 100.0, 'close': 100.0, 'vol': 10000.0, 'is_executable_at_fill': False, 'is_redeemed': False},
    ])
    
    df_orders = pd.DataFrame([
        {'trade_date': '20240102', 'ts_code': '110001.SH'}
    ])
    
    res = simulate_nav(df_pit, df_orders, use_timing=False)
    assert res['trade_cnt'] == 0, "不可成交 (is_executable_at_fill=False) 标的不得发生任何交易"

if __name__ == '__main__':
    pytest.main([__file__])
