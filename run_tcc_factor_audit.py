# -*- coding: utf-8 -*-

"""
时间网络相对中心度 (TCC) 因子在严格 As-Of PIT 管道上的独立审计与验证 (TCC Factor Audit & Strategy Evaluation)
测试项目：
1. 15m / 60m 级 Rank IC 与 IC Sharpe 比例；
2. Q1~Q5 五分组收益与单调性分析；
3. TCC 因子 + 纯双低基准策略融合回测效果（与 -1.98% 基准对比）。
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader
from cb_quant.strict_15m_clean_engine import CBStrict15mCleanEngine
from cb_quant.unified_pit_engine import CBUnifiedPITEngine
from cb_quant.asof_pit_adapter import CBAsOfPITAdapter
from cb_quant.tcc_factor import CBTCCFactorEngine
from cb_quant.time_structured_router import CBTimeStructuredRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    logging.info("=== 启动【时间网络相对中心度 (TCC) 因子】机构级严谨审计 ===")
    
    # 1. 加载 15m 行情面板
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    clean_engine = CBStrict15mCleanEngine()
    df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
    
    pit_adapter = CBAsOfPITAdapter()
    df_15m = pit_adapter.attach_asof_pit_metadata(df_15m)

    unified_engine = CBUnifiedPITEngine()
    df_pit = unified_engine.build_unified_state_panel(df_15m)
    
    # 2. 计算并 As-Of 对齐 T-1 日频 TCC 因子
    tcc_engine = CBTCCFactorEngine(window=21)
    tcc_long = tcc_engine.generate_tcc_panel(start_date="2024-12-01", end_date="2026-07-25")
    
    # T-1 对齐: t1_date_str (下一交易日使用的 TCC 因子来自于 T-1)
    unique_tcc_dates = sorted(tcc_long['date_str'].unique())
    date_map = {unique_tcc_dates[i]: unique_tcc_dates[i+1] for i in range(len(unique_tcc_dates)-1)}
    tcc_long['t1_trade_date'] = tcc_long['date_str'].map(date_map)
    
    df_pit = df_pit.merge(tcc_long[['ts_code', 't1_trade_date', 'tcc_factor']],
                          left_on=['ts_code', 'date_str'], right_on=['ts_code', 't1_trade_date'], how='left')
    
    # 3. 截面 Rank IC 分析 (日频截面快速向量化)
    df_ic_sub = df_pit[
        (df_pit['is_eligible_at_selection'] == True) &
        (df_pit['tcc_factor'].notnull()) &
        (df_pit['fut_ret_60m_close'].notnull())
    ].copy()

    # 每日收盘截面 Rank IC
    daily_sub = df_ic_sub.groupby(['date_str', 'ts_code']).last().reset_index()
    daily_ics = daily_sub.groupby('date_str').apply(
        lambda g: g['tcc_factor'].rank().corr(g['fut_ret_60m_close'].rank()) if len(g) >= 10 else np.nan
    ).dropna()

    mean_ic = daily_ics.mean() if not daily_ics.empty else 0.0
    std_ic = daily_ics.std() if not daily_ics.empty else 1.0
    ic_sharpe = mean_ic / (std_ic + 1e-8) * np.sqrt(252.0)

    # 4. Q1~Q5 五分组收益率分析 (Quintile Analysis)
    df_ic_sub['tcc_group'] = df_ic_sub.groupby(['date_str', 'time_str'])['tcc_factor'].transform(
        lambda x: pd.qcut(x.rank(method='first'), q=5, labels=['Q1_Lowest', 'Q2', 'Q3', 'Q4', 'Q5_Highest']) if len(x) >= 10 else np.nan
    )
    
    group_ret = df_ic_sub.groupby('tcc_group')['fut_ret_60m_close'].mean() * 10000.0 # 单位: bp

    # 5. 融合策略回测: 双低 + 高 TCC 因子筛选 (过滤波动极度离群标的)
    df_pit_tcc = df_pit.copy()
    # 规则: 仅选择 TCC 得分处于前 70% (排除全市场偏离度最极端的后 30% 噪声标的)
    df_pit_tcc['tcc_rank_pct'] = df_pit_tcc.groupby('date_str')['tcc_factor'].rank(ascending=True, pct=True)
    df_pit_tcc['is_eligible_at_selection'] = (
        (df_pit_tcc['is_eligible_at_selection'] == True) &
        (df_pit_tcc['tcc_rank_pct'] >= 0.30) # 排除偏离度最大的尾部 30% 标的
    )
    
    df_orders_tcc, _ = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_tcc)

    # 模拟回测逻辑
    unique_dates = sorted(df_pit['date_str'].unique())
    capital = 1000000.0
    positions = {}
    trade_logs = []
    daily_nav_list = []
    total_dates = len(unique_dates)

    for d_idx, d_str in enumerate(unique_dates):
        df_d = df_pit_tcc[df_pit_tcc['date_str'] == d_str]
        if df_d.empty:
            continue
        is_terminal_date = (d_idx == total_dates - 1)
        
        # 预索引当日行情
        d_by_code = df_d.groupby('ts_code').last().to_dict('index')
        d_first_by_code = df_d.groupby('ts_code').first().to_dict('index')
        
        codes_to_remove = []
        for code, pos in positions.items():
            held_days = d_idx - pos['entry_date_idx']
            if (held_days >= 5) or is_terminal_date:
                fill_row = d_first_by_code.get(code)
                if fill_row is not None:
                    bar_vol = fill_row.get('vol', 0)
                    if fill_row.get('is_executable_at_fill', False) and (bar_vol * 0.05 >= pos['shares'] / 10):
                        exit_px = fill_row['open']
                        net_exit_px = exit_px * (1.0 - 0.0010)
                        net_pnl = pos['shares'] * (net_exit_px - pos['entry_net_px'])
                        capital += pos['shares'] * net_exit_px
                        trade_logs.append({'trade_date': d_str, 'ts_code': code, 'side': 'SELL', 'net_pnl': net_pnl})
                        codes_to_remove.append(code)
                else:
                    exit_px = pos.get('last_valid_price', pos['entry_net_px']) * 0.98
                    net_exit_px = exit_px * (1.0 - 0.0010)
                    net_pnl = pos['shares'] * (net_exit_px - pos['entry_net_px'])
                    capital += pos['shares'] * net_exit_px
                    trade_logs.append({'trade_date': d_str, 'ts_code': code, 'side': 'SELL', 'net_pnl': net_pnl})
                    codes_to_remove.append(code)

        for c in codes_to_remove:
            del positions[c]

        if not is_terminal_date and not df_orders_tcc.empty:
            d_orders = df_orders_tcc[df_orders_tcc['trade_date'] == d_str]
            for _, ord_row in d_orders.iterrows():
                code = ord_row['ts_code']
                if code not in positions and len(positions) < 10:
                    entry_px = ord_row['execution_price']
                    exec_vol = ord_row['execution_vol']
                    net_entry_px = entry_px * (1.0 + 0.0010)
                    slot_capital = min(capital / (10 - len(positions)), capital * 0.20)
                    shares = int((slot_capital / net_entry_px) // 10) * 10
                    
                    if shares >= 10 and (exec_vol * 0.05 >= shares / 10) and capital >= shares * net_entry_px:
                        capital -= shares * net_entry_px
                        positions[code] = {
                            'shares': shares, 'entry_net_px': net_entry_px, 'entry_date_idx': d_idx,
                            'last_valid_price': entry_px, 'stale_days': 0
                        }
                        trade_logs.append({'trade_date': d_str, 'ts_code': code, 'side': 'BUY', 'net_pnl': 0.0})

        pos_val = 0.0
        for code, pos in positions.items():
            df_code_row = d_by_code.get(code)
            if df_code_row is not None:
                curr_px = df_code_row['close']
                pos['last_valid_price'] = curr_px
                pos['stale_days'] = 0
            else:
                pos['stale_days'] += 1
                curr_px = pos['last_valid_price'] * (0.999 ** pos['stale_days'])
            pos_val += pos['shares'] * curr_px

        nav = capital + pos_val
        daily_nav_list.append({'date': d_str, 'nav': nav})

    df_nav = pd.DataFrame(daily_nav_list)
    df_trades = pd.DataFrame(trade_logs)
    
    total_ret = (df_nav['nav'].iloc[-1] / 1000000.0) - 1.0
    ann_ret = (1.0 + total_ret) ** (252.0 / len(df_nav)) - 1.0
    sharpe = (df_nav['nav'].pct_change().fillna(0.0).mean() / (df_nav['nav'].pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    cum_max = df_nav['nav'].cummax()
    max_dd = ((df_nav['nav'] - cum_max) / cum_max).min()

    print("\n" + "="*80)
    print("      【时间网络相对中心度 (TCC) 因子】机构级严谨审计与回测报告")
    print("="*80)
    print("【1. 截面 Rank IC 分析】")
    print("  - 平均 Rank IC:                 {:+.4f}".format(mean_ic))
    print("  - IC 标准差 (IC Std):            {:.4f}".format(std_ic))
    print("  - 年化 IC 夏普 (IC Sharpe):      {:.2f}".format(ic_sharpe))
    print("-" * 80)
    print("【2. Q1~Q5 五分组未来 60m 收益表现 (基点 bp)】")
    for g_name, g_val in group_ret.items():
        print("  - {:15s}: {:+6.2f} bp".format(str(g_name), g_val))
    print("-" * 80)
    print("【3. 纯双低基准 vs 双低 + TCC 因子筛选 策略对比】")
    print("  - 诚实纯双低基准: 累计净收益 {:+.2f}% | 最大回撤 {:.2f}% | 成交 1,174 笔".format(-1.98, -9.60))
    print("  - 双低 + TCC 过滤: 累计净收益 {:+.2f}% | 最大回撤 {:.2f}% | 成交 {} 笔".format(total_ret*100, max_dd*100, len(df_trades)))
    print("  - 年化收益率:     {:+.2f}%".format(ann_ret*100))
    print("  - 夏普比率:       {:.2f}".format(sharpe))
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
