# -*- coding: utf-8 -*-

"""
快速高效：基于诚实新基准的策略升级全流程评估 (Fast Strict Enhanced Backtest Engine)
对比在严格 As-Of PIT 管道上的四种配置：
1. Baseline (纯双低): -1.98%
2. Config 1: 纯双低 + GBDT 机器学习 Alpha 预测
3. Config 2: Config 1 + 智能限价单被动吃价差 (+5 bps)
4. Config 3: Config 2 + 大盘 20MA 择时开关 (死叉空仓)
5. Config 4: 80/20 组合部署框架 (80% 核心层 + 20% 增强层)
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader
from cb_quant.strict_15m_clean_engine import CBStrict15mCleanEngine
from cb_quant.unified_pit_engine import CBUnifiedPITEngine
from cb_quant.asof_pit_adapter import CBAsOfPITAdapter
from cb_quant.time_structured_router import CBTimeStructuredRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def simulate_portfolio_nav(df_pit, df_orders, use_smart_limit=False, use_market_timing=False):
    unique_dates = sorted(df_pit['date_str'].unique())
    daily_market_close = df_pit.groupby('date_str')['close'].mean()
    market_ma20 = daily_market_close.rolling(20, min_periods=5).mean()
    
    capital = 1000000.0
    positions = {}
    trade_logs = []
    daily_nav_list = []
    total_dates = len(unique_dates)

    for d_idx, d_str in enumerate(unique_dates):
        df_d = df_pit[df_pit['date_str'] == d_str]
        if df_d.empty:
            continue
            
        is_terminal_date = (d_idx == total_dates - 1)
        
        mkt_close_t1 = daily_market_close.get(unique_dates[d_idx-1], np.nan) if d_idx > 0 else np.nan
        mkt_ma20_t1 = market_ma20.get(unique_dates[d_idx-1], np.nan) if d_idx > 0 else np.nan
        market_bearish = False
        if use_market_timing and not np.isnan(mkt_close_t1) and not np.isnan(mkt_ma20_t1):
            if mkt_close_t1 < mkt_ma20_t1:
                market_bearish = True

        codes_to_remove = []
        for code, pos in positions.items():
            held_days = d_idx - pos['entry_date_idx']
            should_exit = (held_days >= 5) or is_terminal_date or market_bearish
            
            if should_exit:
                df_code_d = df_d[df_d['ts_code'] == code]
                if not df_code_d.empty:
                    fill_row = df_code_d.iloc[0]
                    bar_vol = fill_row.get('vol', 0)
                    if fill_row.get('is_executable_at_fill', False) and (bar_vol * 0.05 >= pos['shares'] / 10):
                        raw_exit_px = fill_row['open']
                        exit_px = raw_exit_px * 1.0005 if use_smart_limit else raw_exit_px
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

        if not is_terminal_date and not market_bearish and not df_orders.empty:
            d_orders = df_orders[df_orders['trade_date'] == d_str]
            for _, ord_row in d_orders.iterrows():
                code = ord_row['ts_code']
                if code not in positions and len(positions) < 10:
                    raw_entry_px = ord_row['execution_price']
                    exec_vol = ord_row['execution_vol']
                    
                    entry_px = raw_entry_px * 0.9995 if use_smart_limit else raw_entry_px
                    net_entry_px = entry_px * (1.0 + 0.0010)
                    slot_capital = min(capital / (10 - len(positions)), capital * 0.20)
                    shares = int((slot_capital / net_entry_px) // 10) * 10
                    
                    if shares >= 10 and (exec_vol * 0.05 >= shares / 10) and capital >= shares * net_entry_px:
                        capital -= shares * net_entry_px
                        positions[code] = {
                            'shares': shares,
                            'entry_net_px': net_entry_px,
                            'entry_date_idx': d_idx,
                            'last_valid_price': entry_px,
                            'stale_days': 0
                        }
                        trade_logs.append({'trade_date': d_str, 'ts_code': code, 'side': 'BUY', 'net_pnl': 0.0})

        pos_val = 0.0
        for code, pos in positions.items():
            df_code_d = df_d[df_d['ts_code'] == code]
            if not df_code_d.empty:
                curr_px = df_code_d.iloc[-1]['close']
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

    return {
        'total_ret': total_ret, 'ann_ret': ann_ret, 'sharpe': sharpe,
        'max_dd': max_dd, 'trade_cnt': len(df_trades), 'nav_series': df_nav['nav']
    }

def main():
    logging.info("=== 启动【快速高效 策略升级全评估】 ===")
    
    # 1. 一次性加载全量 15m PIT 面板
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    clean_engine = CBStrict15mCleanEngine()
    df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
    
    pit_adapter = CBAsOfPITAdapter()
    df_15m = pit_adapter.attach_asof_pit_metadata(df_15m)

    unified_engine = CBUnifiedPITEngine()
    df_pit_base = unified_engine.build_unified_state_panel(df_15m)
    
    # 0. 纯双低基准 (Baseline)
    df_orders_base, _ = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_base)
    res_base = simulate_portfolio_nav(df_pit_base, df_orders_base, use_smart_limit=False, use_market_timing=False)

    # 1. GBDT ML Alpha 增强 (Config 1)
    df_pit_ml = df_pit_base.copy()
    model_path = "lgb_strict_pit_model.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        feature_cols = [
            'double_low', 'premium_rate_t1', 'conv_value_t1', 'curr_iss_amt',
            'chip_profit_ratio', 'chip_concentration_90', 'chip_position_20d',
            'spike_ratio', 'vol', 'amount'
        ]
        for c in feature_cols:
            if c not in df_pit_ml.columns:
                df_pit_ml[c] = np.nan
        df_pit_clean = df_pit_ml[feature_cols].fillna(0.0)
        df_pit_ml['ml_pred'] = model.predict(df_pit_clean)
        df_pit_ml['score_rank'] = df_pit_ml.groupby('date_str')['double_low'].rank(ascending=True, method='min') - \
                                  df_pit_ml.groupby('date_str')['ml_pred'].rank(ascending=False, method='min') * 2.0
        df_pit_ml['double_low'] = df_pit_ml['score_rank']

    df_orders_ml, _ = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_ml)
    res_cfg1 = simulate_portfolio_nav(df_pit_ml, df_orders_ml, use_smart_limit=False, use_market_timing=False)

    # 2. Config 2: ML Alpha + 智能限价 (+5 bps)
    res_cfg2 = simulate_portfolio_nav(df_pit_ml, df_orders_ml, use_smart_limit=True, use_market_timing=False)

    # 3. Config 3: ML Alpha + 智能限价 + 20MA 择时
    res_cfg3 = simulate_portfolio_nav(df_pit_ml, df_orders_ml, use_smart_limit=True, use_market_timing=True)

    # 4. Config 4: 80/20 组合部署框架 (80% 核心层 + 20% 增强层)
    nav_portfolio = 0.80 * res_base['nav_series'] + 0.20 * res_cfg2['nav_series']
    tot_port = (nav_portfolio.iloc[-1] / 1000000.0) - 1.0
    ann_port = (1.0 + tot_port) ** (252.0 / len(nav_portfolio)) - 1.0
    sharpe_port = (nav_portfolio.pct_change().fillna(0.0).mean() / (nav_portfolio.pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    cum_max = nav_portfolio.cummax()
    max_dd_port = ((nav_portfolio - cum_max) / cum_max).min()

    print("\n" + "="*85)
    print("         【基于诚实新基准 (-1.98%) 的策略升级与 80/20 组合对比报告】")
    print("="*85)
    print("配置名称                         | 累计收益率 | 年化收益率 | 夏普比率 | 最大回撤 | 总成交笔数")
    print("-" * 85)
    print("0. 诚实新基准 (纯双低)           | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  | {:6d} 笔".format(
        res_base['total_ret']*100, res_base['ann_ret']*100, res_base['sharpe'], res_base['max_dd']*100, res_base['trade_cnt']))
    print("1. 双低 + GBDT ML Alpha 增强     | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  | {:6d} 笔".format(
        res_cfg1['total_ret']*100, res_cfg1['ann_ret']*100, res_cfg1['sharpe'], res_cfg1['max_dd']*100, res_cfg1['trade_cnt']))
    print("2. ML Alpha + 智能限价 (+5bp)    | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  | {:6d} 笔".format(
        res_cfg2['total_ret']*100, res_cfg2['ann_ret']*100, res_cfg2['sharpe'], res_cfg2['max_dd']*100, res_cfg2['trade_cnt']))
    print("3. ML Alpha + 智能限价 + 20MA择时| {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  | {:6d} 笔".format(
        res_cfg3['total_ret']*100, res_cfg3['ann_ret']*100, res_cfg3['sharpe'], res_cfg3['max_dd']*100, res_cfg3['trade_cnt']))
    print("4. 80/20 组合部署框架 (最终推荐) | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  |  --   ".format(
        tot_port*100, ann_port*100, sharpe_port, max_dd_port*100))
    print("="*85 + "\n")

if __name__ == '__main__':
    main()
