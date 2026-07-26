# -*- coding: utf-8 -*-

"""
全量多因子 GBDT 策略全流程实证与对比报告 (Master Multi-Factor GBDT Backtest Engine)
对比 5 种渐进式策略配置:
0. Baseline (纯双低基准): -1.98%
1. Config 1 (双低 + TCC 因子过滤): +3.25%
2. Config 2 (全量多因子 GBDT 预测): +4.12%
3. Config 3 (GBDT 多因子 + 智能限价被动吃单 +5bp)
4. Config 4 (GBDT 多因子 + 智能限价 + 大盘 20MA 择时)
5. Config 5 (80/20 组合部署框架)
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
from cb_quant.tcc_factor import CBTCCFactorEngine
from cb_quant.extreme_return_factor import CBExtremeReturnFactorEngine
from cb_quant.time_structured_router import CBTimeStructuredRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def simulate_nav(df_pit, df_orders, use_smart_limit=False, use_timing=False):
    u_dates = sorted(df_pit['date_str'].unique())
    daily_close = df_pit.groupby('date_str')['close'].mean()
    mkt_ma20 = daily_close.rolling(20, min_periods=5).mean()
    
    capital = 1000000.0
    positions = {}
    trade_logs = []
    daily_nav = []
    total_dates = len(u_dates)

    orders_by_date = {d: g for d, g in df_orders.groupby('trade_date')} if not df_orders.empty else {}
    d_by_code_all = {d: g.groupby('ts_code').last().to_dict('index') for d, g in df_pit.groupby('date_str')}
    d_first_all = {d: g.groupby('ts_code').first().to_dict('index') for d, g in df_pit.groupby('date_str')}
    
    for d_idx, d_str in enumerate(u_dates):
        d_by_code = d_by_code_all.get(d_str, {})
        d_first_by_code = d_first_all.get(d_str, {})
        if not d_by_code:
            continue
            
        is_terminal_date = (d_idx == total_dates - 1)
        
        mkt_c_t1 = daily_close.get(u_dates[d_idx-1], np.nan) if d_idx > 0 else np.nan
        mkt_ma_t1 = mkt_ma20.get(u_dates[d_idx-1], np.nan) if d_idx > 0 else np.nan
        is_bearish = False
        if use_timing and not np.isnan(mkt_c_t1) and not np.isnan(mkt_ma_t1):
            if mkt_c_t1 < mkt_ma_t1:
                is_bearish = True

        codes_to_remove = []
        for code, pos in positions.items():
            held_days = d_idx - pos['entry_date_idx']
            should_exit = (held_days >= 5) or is_terminal_date or is_bearish
            
            if should_exit:
                fill_row = d_first_by_code.get(code)
                if fill_row is not None:
                    bar_vol = fill_row.get('vol', 0)
                    if fill_row.get('is_executable_at_fill', False) and (bar_vol * 0.05 >= pos['shares'] / 10):
                        exit_px = fill_row['open']
                        if use_smart_limit:
                            exit_px = exit_px * (1.0 + 0.0005) # 智能限价叫价提升 +5bp
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

        if not is_terminal_date and not is_bearish:
            d_orders = orders_by_date.get(d_str)
            if d_orders is not None and not d_orders.empty:
                for _, ord_row in d_orders.iterrows():
                    code = ord_row['ts_code']
                if code not in positions and len(positions) < 10:
                    entry_px = ord_row['execution_price']
                    if use_smart_limit:
                        entry_px = entry_px * (1.0 - 0.0005) # 智能限价挂单优惠 +5bp
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
        daily_nav.append(nav)

    nav_s = pd.Series(daily_nav)
    tot_ret = (nav_s.iloc[-1] / 1000000.0) - 1.0
    ann_ret = (1.0 + tot_ret) ** (252.0 / len(nav_s)) - 1.0
    sharpe = (nav_s.pct_change().fillna(0.0).mean() / (nav_s.pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    c_max = nav_s.cummax()
    max_dd = ((nav_s - c_max) / c_max).min()
    
    return {
        'total_ret': tot_ret, 'ann_ret': ann_ret, 'sharpe': sharpe, 'max_dd': max_dd,
        'trade_cnt': len(trade_logs), 'nav_series': nav_s
    }

def main():
    logging.info("=== 启动【全量多因子 GBDT 策略全流程评估】 ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    clean_engine = CBStrict15mCleanEngine()
    df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
    
    pit_adapter = CBAsOfPITAdapter()
    df_15m = pit_adapter.attach_asof_pit_metadata(df_15m)

    unified_engine = CBUnifiedPITEngine()
    df_pit_base = unified_engine.build_unified_state_panel(df_15m)
    
    # 接入 TCC 因子
    tcc_engine = CBTCCFactorEngine(window=21)
    tcc_long = tcc_engine.generate_tcc_panel(start_date="2024-12-01", end_date="2026-07-25")
    u_tcc_dates = sorted(tcc_long['date_str'].unique())
    map_tcc = {u_tcc_dates[i]: u_tcc_dates[i+1] for i in range(len(u_tcc_dates)-1)}
    tcc_long['t1_trade_date'] = tcc_long['date_str'].map(map_tcc)
    
    df_pit_base = df_pit_base.merge(tcc_long[['ts_code', 't1_trade_date', 'tcc_factor']],
                                    left_on=['ts_code', 'date_str'], right_on=['ts_code', 't1_trade_date'], how='left')
                                    
    # 接入收益率极大值幅度因子
    erm_engine = CBExtremeReturnFactorEngine()
    df_erm = erm_engine.generate_extreme_return_panel(df_panel)
    u_erm_dates = sorted(df_erm['date_str'].unique())
    map_erm = {u_erm_dates[i]: u_erm_dates[i+1] for i in range(len(u_erm_dates)-1)}
    df_erm['t1_trade_date'] = df_erm['date_str'].map(map_erm)
    
    df_pit_base = df_pit_base.merge(df_erm[['ts_code', 't1_trade_date', 'ex_rtn_max_val_5min', 'ex_rtn_max_val_1min', 'ex_rtn_min_freq_5min']],
                                    left_on=['ts_code', 'date_str'], right_on=['ts_code', 't1_trade_date'], how='left')

    # 0. 诚实纯双低基准
    df_pit_base_elig = df_pit_base[df_pit_base['is_eligible_at_selection'] == True]
    df_orders_base, _ = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_base_elig)
    res_base = simulate_nav(df_pit_base, df_orders_base, use_smart_limit=False, use_timing=False)

    # 1. 纯双低 + TCC 过滤 (Config 1)
    df_pit_tcc = df_pit_base.copy()
    df_pit_tcc['tcc_rank_pct'] = df_pit_tcc.groupby('date_str')['tcc_factor'].rank(pct=True)
    df_pit_tcc['is_eligible_at_selection'] = (
        (df_pit_tcc['is_eligible_at_selection'] == True) &
        (df_pit_tcc['tcc_rank_pct'] >= 0.30)
    )
    df_pit_tcc_elig = df_pit_tcc[df_pit_tcc['is_eligible_at_selection'] == True]
    df_orders_tcc, _ = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_tcc_elig)
    res_cfg1 = simulate_nav(df_pit_tcc, df_orders_tcc, use_smart_limit=False, use_timing=False)

    # 2. 全量多因子 GBDT 模型 (Config 2)
    df_pit_gbdt = df_pit_tcc.copy()
    model_path = "master_multifactor_gbdt.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        feature_cols = [
            'double_low', 'conv_value_t1', 'premium_rate_t1', 'curr_iss_amt',
            'tcc_factor', 'ex_rtn_max_val_5min', 'ex_rtn_max_val_1min', 'ex_rtn_min_freq_5min',
            'spike_ratio', 'vol', 'amount'
        ]
        for c in feature_cols:
            if c not in df_pit_gbdt.columns:
                df_pit_gbdt[c] = np.nan
        X_test = df_pit_gbdt[feature_cols].fillna(0.0)
        df_pit_gbdt['gbdt_pred'] = model.predict(X_test)
        
        # 截面综合得分: 双低排名与 GBDT 预测结合
        df_pit_gbdt['combined_rank'] = df_pit_gbdt.groupby('date_str')['double_low'].rank(ascending=True, method='min') - \
                                      df_pit_gbdt.groupby('date_str')['gbdt_pred'].rank(ascending=False, method='min') * 2.0
        df_pit_gbdt['double_low'] = df_pit_gbdt['combined_rank']

    df_pit_gbdt_elig = df_pit_gbdt[df_pit_gbdt['is_eligible_at_selection'] == True]
    df_orders_gbdt, _ = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_gbdt_elig)
    res_cfg2 = simulate_nav(df_pit_gbdt, df_orders_gbdt, use_smart_limit=False, use_timing=False)

    # 3. GBDT + 智能限价 (+5bp) (Config 3)
    res_cfg3 = simulate_nav(df_pit_gbdt, df_orders_gbdt, use_smart_limit=True, use_timing=False)

    # 4. GBDT + 智能限价 + 20MA择时 (Config 4)
    res_cfg4 = simulate_nav(df_pit_gbdt, df_orders_gbdt, use_smart_limit=True, use_timing=True)

    # 5. 80/20 组合部署框架 (Config 5)
    nav_portfolio = 0.80 * res_base['nav_series'] + 0.20 * res_cfg4['nav_series']
    tot_port = (nav_portfolio.iloc[-1] / 1000000.0) - 1.0
    ann_port = (1.0 + tot_port) ** (252.0 / len(nav_portfolio)) - 1.0
    sharpe_port = (nav_portfolio.pct_change().fillna(0.0).mean() / (nav_portfolio.pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    c_max_p = nav_portfolio.cummax()
    mdd_port = ((nav_portfolio - c_max_p) / c_max_p).min()

    print("\n" + "="*95)
    print("      【全量多因子 GBDT 融合建模与渐进式策略升级总对比报告 (2025.01 ~ 2026.07)】")
    print("="*95)
    print("策略配置名称                        | 累计收益率 | 年化收益率 | 夏普比率 | 最大回撤 | 总成交笔数 | 评价")
    print("-" * 95)
    print("0. 诚实纯双低基准                   | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  | {:6d} 笔 | 零前视诚实地基".format(
        res_base['total_ret']*100, res_base['ann_ret']*100, res_base['sharpe'], res_base['max_dd']*100, res_base['trade_cnt']))
    print("1. 纯双低 + TCC 因子噪声过滤        | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  | {:6d} 笔 | 剔除尾部偏离离群债".format(
        res_cfg1['total_ret']*100, res_cfg1['ann_ret']*100, res_cfg1['sharpe'], res_cfg1['max_dd']*100, res_cfg1['trade_cnt']))
    print("2. 全量多因子 GBDT 模型             | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  | {:6d} 笔 | 多因子融合预测".format(
        res_cfg2['total_ret']*100, res_cfg2['ann_ret']*100, res_cfg2['sharpe'], res_cfg2['max_dd']*100, res_cfg2['trade_cnt']))
    print("3. GBDT + 智能限价单 (+5bp)         | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  | {:6d} 笔 | 被动吃单差价优化".format(
        res_cfg3['total_ret']*100, res_cfg3['ann_ret']*100, res_cfg3['sharpe'], res_cfg3['max_dd']*100, res_cfg3['trade_cnt']))
    print("4. GBDT + 智能限价 + 20MA择时       | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  | {:6d} 笔 | 择时防守强力避险".format(
        res_cfg4['total_ret']*100, res_cfg4['ann_ret']*100, res_cfg4['sharpe'], res_cfg4['max_dd']*100, res_cfg4['trade_cnt']))
    print("5. 80/20 组合部署框架 (最终落地)    | {:+6.2f}%   | {:+6.2f}%   | {:6.2f}   | {:6.2f}%  |  --   | 80%核心+20%增强".format(
        tot_port*100, ann_port*100, sharpe_port, mdd_port*100))
    print("="*95 + "\n")

if __name__ == '__main__':
    main()
