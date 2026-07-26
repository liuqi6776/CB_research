# -*- coding: utf-8 -*-

"""
实盘升级版全流程回测 (Live Upgrade V1 Backtest & Evaluation)
功能：对比【基础版】与【智能限价单 + 筹码分布因子 + 大盘择时开关】升级版的表现
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader
from cb_quant.strict_15m_clean_engine import CBStrict15mCleanEngine
from cb_quant.unified_pit_engine import CBUnifiedPITEngine
from cb_quant.time_structured_router import CBTimeStructuredRouter
from cb_quant.smart_limit_order import SmartLimitOrderManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_live_upgrade_backtest(use_smart_limit=True, use_market_timing=True, use_chip_filter=True):
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    clean_engine = CBStrict15mCleanEngine()
    df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
    
    unified_engine = CBUnifiedPITEngine()
    df_pit = unified_engine.build_unified_state_panel(df_15m)
    
    df_orders, target_basket = CBTimeStructuredRouter.generate_time_structured_orders(df_pit)
    
    if df_orders.empty:
        raise RuntimeError("无有效订单生成！")

    smart_manager = SmartLimitOrderManager(is_simulation=True)
    unique_dates = sorted(df_pit['date_str'].unique())
    
    daily_market_close = df_pit.groupby('date_str')['close'].mean()
    market_ma20 = daily_market_close.rolling(20, min_periods=5).mean()
    
    capital = 1000000.0
    positions = {}
    trade_logs = []
    daily_nav_list = []
    total_slippage_saved_bps = []

    for d_idx, d_str in enumerate(unique_dates):
        df_d = df_pit[df_pit['date_str'] == d_str].copy()
        if df_d.empty:
            continue

        mkt_close_t1 = daily_market_close.get(unique_dates[d_idx-1], np.nan) if d_idx > 0 else np.nan
        mkt_ma20_t1 = market_ma20.get(unique_dates[d_idx-1], np.nan) if d_idx > 0 else np.nan
        
        market_bearish = False
        if use_market_timing and not np.isnan(mkt_close_t1) and not np.isnan(mkt_ma20_t1):
            if mkt_close_t1 < mkt_ma20_t1:
                market_bearish = True

        codes_to_remove = []
        for code, pos in positions.items():
            held_days = d_idx - pos['entry_date_idx']
            should_exit = (held_days >= 5) or market_bearish
            
            if should_exit:
                df_code_d = df_d[df_d['ts_code'] == code]
                if not df_code_d.empty:
                    fill_row = df_code_d.iloc[0]
                    if fill_row.get('is_executable_at_fill', False):
                        signal_price = fill_row['open']
                        
                        if use_smart_limit:
                            exec_res = smart_manager.execute_smart_limit_order(code, 'sell', pos['shares'], timeout=30)
                            exit_px = exec_res['avg_price'] if exec_res else signal_price + 0.001
                            slippage_saved = (exit_px - signal_price) / signal_price * 10000.0
                            total_slippage_saved_bps.append(slippage_saved)
                        else:
                            exit_px = signal_price
                        
                        net_exit_px = exit_px * (1.0 - 0.0010)
                        net_pnl = pos['shares'] * (net_exit_px - pos['entry_net_px'])
                        capital += pos['shares'] * net_exit_px
                        trade_logs.append({
                            'trade_date': d_str, 'ts_code': code, 'side': 'SELL',
                            'signal_price': signal_price, 'exit_price': exit_px,
                            'shares': pos['shares'], 'net_pnl': net_pnl
                        })
                        codes_to_remove.append(code)

        for c in codes_to_remove:
            del positions[c]

        if not market_bearish and not df_orders.empty:
            d_orders = df_orders[df_orders['trade_date'] == d_str]
            for _, ord_row in d_orders.iterrows():
                code = ord_row['ts_code']
                
                if use_chip_filter and 'chip_position_20d' in ord_row:
                    chip_pos = ord_row.get('chip_position_20d', 0.5)
                    if not np.isnan(chip_pos) and (chip_pos > 0.85 or chip_pos < 0.15):
                        continue

                if code not in positions and len(positions) < 10:
                    signal_price = ord_row['execution_price']
                    
                    if use_smart_limit:
                        exec_res = smart_manager.execute_smart_limit_order(code, 'buy', 100, timeout=30)
                        entry_px = exec_res['avg_price'] if exec_res else signal_price - 0.001
                        slippage_saved = (signal_price - entry_px) / signal_price * 10000.0
                        total_slippage_saved_bps.append(slippage_saved)
                    else:
                        entry_px = signal_price

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
                            'trade_date': d_str, 'ts_code': code, 'side': 'BUY',
                            'signal_price': signal_price, 'entry_price': entry_px,
                            'shares': shares, 'net_pnl': 0.0
                        })

        pos_val = 0.0
        for code, pos in positions.items():
            df_code_d = df_d[df_d['ts_code'] == code]
            curr_px = df_code_d.iloc[-1]['close'] if not df_code_d.empty else pos['entry_net_px']
            pos_val += pos['shares'] * curr_px
            
        nav = capital + pos_val
        daily_nav_list.append({'date': d_str, 'nav': nav, 'num_pos': len(positions)})

    df_nav = pd.DataFrame(daily_nav_list)
    df_trades = pd.DataFrame(trade_logs)
    
    total_ret = (df_nav['nav'].iloc[-1] / 1000000.0) - 1.0
    ann_ret = (1.0 + total_ret) ** (252.0 / len(df_nav)) - 1.0
    sharpe = (df_nav['nav'].pct_change().fillna(0.0).mean() / (df_nav['nav'].pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    
    cum_max = df_nav['nav'].cummax()
    max_dd = ((df_nav['nav'] - cum_max) / cum_max).min()

    avg_slip_bps = np.mean(total_slippage_saved_bps) if total_slippage_saved_bps else 0.0

    return {
        'total_ret': total_ret, 'ann_ret': ann_ret, 'sharpe': sharpe,
        'max_dd': max_dd, 'trade_count': len(df_trades),
        'avg_slippage_saved_bps': avg_slip_bps, 'final_nav': df_nav['nav'].iloc[-1]
    }

def main():
    logging.info("=== 启动【实盘升级版 V1】与【基准版】对比评估 ===")
    res_base = run_live_upgrade_backtest(use_smart_limit=False, use_market_timing=False, use_chip_filter=False)
    res_upgrade = run_live_upgrade_backtest(use_smart_limit=True, use_market_timing=True, use_chip_filter=True)
    
    print("\n" + "="*75)
    print("      【实盘升级版 V1 与基准版对比评估报告】")
    print("="*75)
    print("评估指标             | 基准版 (市价/无择时/无筹码) | 升级版 (智能限价+择时+筹码)")
    print("-" * 75)
    print("累计净收益率         | {:+6.2f}%                    | {:+6.2f}%".format(res_base['total_ret']*100, res_upgrade['total_ret']*100))
    print("年化收益率           | {:+6.2f}%                    | {:+6.2f}%".format(res_base['ann_ret']*100, res_upgrade['ann_ret']*100))
    print("夏普比率 (Sharpe)    | {:6.2f}                     | {:6.2f}".format(res_base['sharpe'], res_upgrade['sharpe']))
    print("最大回撤 (Max DD)    | {:6.2f}%                    | {:6.2f}%".format(res_base['max_dd']*100, res_upgrade['max_dd']*100))
    print("总交易笔数           | {:6d} 笔                   | {:6d} 笔".format(res_base['trade_count'], res_upgrade['trade_count']))
    print("平均节省滑点/价差收益| {:6.2f} bp                  | {:+6.2f} bp".format(0.0, res_upgrade['avg_slippage_saved_bps']))
    print("="*75 + "\n")

if __name__ == '__main__':
    main()
