# -*- coding: utf-8 -*-

"""
日期化 PIT As-Of Join 全量回测主程序 (彻底废除旧收益结论，在全量 PIT 元数据下重跑基线与消融)
Master Runner for Date-Indexed PIT As-Of Joined Baseline & Ablation Backtest
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader
from cb_quant.strict_15m_clean_engine import CBStrict15mCleanEngine
from cb_quant.pit_metadata_engine import CBPITMetadataEngine
from cb_quant.daily_factor_engine import CBDailyFactorEngine
from cb_quant.intraday_timing_engine import CBIntradayTimingEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    logging.info("=== 启动【日期化 PIT As-Of Join】全量基线与消融回测 ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    # 1. 先聚合为 15m K 线
    clean_engine = CBStrict15mCleanEngine()
    df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
    
    # 2. 应用日期化 PIT 元数据 As-Of Join (强赎日期、退市日期、正股 T-1 报价)
    pit_engine = CBPITMetadataEngine()
    df_pit = pit_engine.apply_pit_asof_join(df_15m)
    
    # 3. 运行 Arm A: 纯日频双低基线 (PIT As-Of Join)
    daily_engine = CBDailyFactorEngine()
    target_basket = daily_engine.compute_daily_selection_panel(df_pit)
    
    # 4. 运行 Arm B: 日频双低 + 15m 盘中择时与陷阱过滤
    timing_signals = CBIntradayTimingEngine.apply_intraday_entry_timing(df_pit, target_basket)
    
    # 5. 低换手投资组合模拟 (持仓 5 天，20bp 摩擦)
    unique_dates = sorted(df_pit['date_str'].unique())
    capital = 1000000.0
    positions = {}
    trade_logs = []
    daily_nav_list = []
    
    timing_set = set(zip(timing_signals['ts_code'], timing_signals['date_str']))
    
    for d_idx, d_str in enumerate(unique_dates):
        df_d = df_pit[df_pit['date_str'] == d_str].copy()
        if df_d.empty:
            continue
            
        t_first = df_d.iloc[0]['trade_time']
        
        # A. 检查止盈平仓
        codes_to_remove = []
        for code, pos in positions.items():
            held_days = d_idx - pos['entry_date_idx']
            if held_days >= 5:
                df_code_d = df_d[df_d['ts_code'] == code]
                if not df_code_d.empty:
                    exit_px = df_code_d.iloc[0]['open']
                    is_tr = df_code_d.iloc[0].get('is_tradable', False)
                    if exit_px > 0 and is_tr:
                        net_exit_px = exit_px * (1.0 - 0.0010)
                        net_pnl = pos['shares'] * (net_exit_px - pos['entry_net_px'])
                        capital += pos['shares'] * net_exit_px
                        trade_logs.append({
                            'trade_time': str(t_first), 'ts_code': code, 'side': 'SELL',
                            'gross_price': exit_px, 'shares': pos['shares'], 'net_pnl': net_pnl
                        })
                        codes_to_remove.append(code)
                        
        for c in codes_to_remove:
            del positions[c]
            
        # B. 检查买入入场
        for code in df_d['ts_code'].unique():
            if (code, d_str) in timing_set and code not in positions:
                if len(positions) < 10:
                    df_code_t = df_d[df_d['ts_code'] == code]
                    if not df_code_t.empty:
                        entry_px = df_code_t.iloc[0]['open']
                        is_tr = df_code_t.iloc[0].get('is_tradable', False)
                        if entry_px > 0 and is_tr:
                            net_entry_px = entry_px * (1.0 + 0.0010)
                            slot_capital = min(capital / (10 - len(positions)), capital * 0.20)
                            shares = int((slot_capital / net_entry_px) // 10) * 10
                            
                            if shares >= 10 and capital >= shares * net_entry_px:
                                capital -= shares * net_entry_px
                                positions[code] = {
                                    'shares': shares,
                                    'entry_net_px': net_entry_px,
                                    'entry_date_idx': d_idx
                                }
                                trade_logs.append({
                                    'trade_time': str(t_first), 'ts_code': code, 'side': 'BUY',
                                    'gross_price': entry_px, 'shares': shares, 'net_pnl': 0.0
                                })

        # C. 结算当日 NAV
        pos_val = 0.0
        for code, pos in positions.items():
            df_code_d = df_d[df_d['ts_code'] == code]
            curr_px = df_code_d.iloc[-1]['close'] if not df_code_d.empty else pos['entry_net_px']
            pos_val += pos['shares'] * curr_px
            
        nav = capital + pos_val
        daily_nav_list.append({'date': d_str, 'nav': nav, 'num_pos': len(positions)})

    df_nav = pd.DataFrame(daily_nav_list)
    df_trades = pd.DataFrame(trade_logs)
    
    df_nav.to_csv("pit_asof_daily_nav.csv", index=False, encoding="utf-8-sig")
    df_trades.to_csv("pit_asof_trade_logs.csv", index=False, encoding="utf-8-sig")

    df_nav['ret'] = df_nav['nav'].pct_change().fillna(0.0)
    total_ret = (df_nav['nav'].iloc[-1] / 1000000.0) - 1.0
    ann_ret = (1.0 + total_ret) ** (252.0 / len(df_nav)) - 1.0
    sharpe = (df_nav['ret'].mean() / (df_nav['ret'].std() + 1e-8)) * np.sqrt(252.0)
    
    cum_max = df_nav['nav'].cummax()
    drawdown = (df_nav['nav'] - cum_max) / cum_max
    max_dd = drawdown.min()
    
    if not df_trades.empty and 'side' in df_trades.columns:
        sells = df_trades[df_trades['side'] == 'SELL']
        win_rate = (sells['net_pnl'] > 0).mean() if not sells.empty else 0.0
    else:
        sells = pd.DataFrame()
        win_rate = 0.0

    print("\n" + "="*75)
    print("      【日期化 PIT As-Of Join】全量基线与消融回测报告")
    print("="*75)
    print("PIT 元数据处理:     As-Of Join 强赎日期 (cb_call_history) + 真实正股 T-1 报价")
    print("零缺省放行规则:     缺失正股/规模/强赎状态时强制 is_tradable = False")
    print("-" * 75)
    print("【PIT 真实数据下可信绩效表现 (2025.01 ~ 2026.07)】")
    print("  - 初始资金:                     1,000,000.00 元")
    print("  - 最终资金:                     {:,.2f} 元".format(df_nav['nav'].iloc[-1]))
    print("  - PIT 真实累计净收益率:        {:+.2f}%".format(total_ret * 100))
    print("  - PIT 真实年化收益率:           {:+.2f}%".format(ann_ret * 100))
    print("  - PIT 真实夏普比率:             {:.2f}".format(sharpe))
    print("  - PIT 真实最大回撤:             {:.2f}%".format(max_dd * 100))
    print("  - 平仓交易总笔数:               {} 笔".format(len(sells)))
    print("  - 胜率 (Win Rate):              {:.1f}%".format(win_rate * 100))
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
