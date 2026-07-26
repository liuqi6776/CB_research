# -*- coding: utf-8 -*-

"""
机构级严谨时间结构管道与 10 级全诊断主程序 (Master Diagnostic & New Exploration Runner)
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    logging.info("=== 启动【机构级严谨时间结构管道与 10 级全诊断】回测程序 ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    # 1. 先聚合为 15m K 线
    clean_engine = CBStrict15mCleanEngine()
    df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
    
    # 2. 建立统一资格状态机与 PIT 引擎 (无任何 .fillna 补齐)
    unified_engine = CBUnifiedPITEngine()
    df_pit = unified_engine.build_unified_state_panel(df_15m)
    
    # 3. 运行时间结构路由器 (生成完整时间戳订单: feature_date < trade_date 且 signal_time < execution_time)
    df_orders, target_basket = CBTimeStructuredRouter.generate_time_structured_orders(df_pit)
    
    # 4. 逐字段 PIT 覆盖率诊断 (Per-Field PIT Coverage Breakdown)
    total_rows = len(df_pit)
    stk_t1_cov = df_pit['stk_close_t1'].notnull().mean() * 100
    conv_px_cov = df_pit['conv_price'].notnull().mean() * 100
    conv_val_cov = df_pit['conv_value_t1'].notnull().mean() * 100
    iss_amt_cov = df_pit['curr_iss_amt'].notnull().mean() * 100
    redeem_cov = df_pit['is_redeemed'].notnull().mean() * 100
    eligible_ratio = df_pit['is_eligible_at_selection'].mean() * 100

    # 5. 统计 10 级全诊断流向数据 (Diagnostic Pipeline Breakdown)
    unique_dates = sorted(df_pit['date_str'].unique())
    diag_rows = []
    
    for d_str in unique_dates:
        df_d = df_pit[df_pit['date_str'] == d_str]
        raw_bonds = df_d['ts_code'].nunique()
        pit_complete = df_d[df_d['conv_value_t1'].notnull() & df_d['curr_iss_amt'].notnull()]['ts_code'].nunique()
        eligible_bonds = df_d[df_d['is_eligible_at_selection'] == True]['ts_code'].nunique()
        top_n = target_basket[target_basket['trade_date'] == d_str]['ts_code'].nunique() if not target_basket.empty else 0
        signals = df_d[df_d['is_executable_at_signal'] == True]['ts_code'].nunique()
        order_cnt = df_orders[df_orders['trade_date'] == d_str]['ts_code'].nunique() if not df_orders.empty else 0
        
        diag_rows.append({
            'trade_date': d_str,
            'raw_bonds': raw_bonds,
            'pit_complete': pit_complete,
            'eligible_bonds': eligible_bonds,
            'top_n_selected': top_n,
            'intraday_signals': signals,
            'orders_generated': order_cnt
        })

    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv("strict_pipeline_diagnostics.csv", index=False, encoding="utf-8-sig")

    # 6. 审计熔断检查：零 PIT、零 eligible、零订单时直接报错退出！
    if df_pit['conv_value_t1'].notnull().sum() == 0:
        raise RuntimeError("【审计熔断触发】PIT 覆盖率为 0，禁止输出平坦 NAV 假结论！")
    if df_pit['is_eligible_at_selection'].sum() == 0:
        raise RuntimeError("【审计熔断触发】T-1 选债合格标的数为 0，禁止输出平坦 NAV 假结论！")
    if df_orders.empty:
        raise RuntimeError("【审计熔断触发】生成的时间结构订单数为 0，禁止输出平坦 NAV 假结论！")

    # 7. 组合模拟 (持仓 5 个交易日，扣除 20bp 摩擦，库存绝对守恒)
    capital = 1000000.0
    positions = {}
    trade_logs = []
    daily_nav_list = []
    order_rejections = []
    
    for d_idx, d_str in enumerate(unique_dates):
        df_d = df_pit[df_pit['date_str'] == d_str].copy()
        if df_d.empty:
            continue
            
        t_first = df_d.iloc[0]['trade_time']
        
        # A. 检查止盈/满期平仓
        codes_to_remove = []
        for code, pos in positions.items():
            held_days = d_idx - pos['entry_date_idx']
            if held_days >= 5:
                df_code_d = df_d[df_d['ts_code'] == code]
                if not df_code_d.empty:
                    fill_row = df_code_d.iloc[0]
                    if fill_row.get('is_executable_at_fill', False):
                        exit_px = fill_row['open']
                        net_exit_px = exit_px * (1.0 - 0.0010)
                        net_pnl = pos['shares'] * (net_exit_px - pos['entry_net_px'])
                        capital += pos['shares'] * net_exit_px
                        trade_logs.append({
                            'trade_date': d_str, 'execution_time': str(fill_row['trade_time']),
                            'ts_code': code, 'side': 'SELL', 'gross_price': exit_px,
                            'shares': pos['shares'], 'net_pnl': net_pnl
                        })
                        codes_to_remove.append(code)
                    else:
                        order_rejections.append({'trade_date': d_str, 'ts_code': code, 'reason': 'Fill bar untradable/halted on SELL'})
                else:
                    order_rejections.append({'trade_date': d_str, 'ts_code': code, 'reason': 'Missing price bar on SELL (Stale)'})
                    
        for c in codes_to_remove:
            del positions[c]
            
        # B. 检查买入入场 (严格从 df_orders 提取订单)
        if not df_orders.empty:
            d_orders = df_orders[df_orders['trade_date'] == d_str]
            for _, ord_row in d_orders.iterrows():
                code = ord_row['ts_code']
                if code not in positions and len(positions) < 10:
                    entry_px = ord_row['execution_price']
                    exec_time = ord_row['execution_time']
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
                            'trade_date': d_str, 'execution_time': exec_time,
                            'ts_code': code, 'side': 'BUY', 'gross_price': entry_px,
                            'shares': shares, 'net_pnl': 0.0
                        })

        # C. 结算当日 NAV (估值守恒)
        pos_val = 0.0
        for code, pos in positions.items():
            df_code_d = df_d[df_d['ts_code'] == code]
            curr_px = df_code_d.iloc[-1]['close'] if not df_code_d.empty else pos['entry_net_px']
            pos_val += pos['shares'] * curr_px
            
        nav = capital + pos_val
        daily_nav_list.append({'date': d_str, 'nav': nav, 'num_pos': len(positions)})

    df_nav = pd.DataFrame(daily_nav_list)
    df_trades = pd.DataFrame(trade_logs)
    
    df_nav.to_csv("strict_pipeline_daily_nav.csv", index=False, encoding="utf-8-sig")
    df_trades.to_csv("strict_pipeline_trades.csv", index=False, encoding="utf-8-sig")

    # 绩效计算
    total_ret = (df_nav['nav'].iloc[-1] / 1000000.0) - 1.0
    ann_ret = (1.0 + total_ret) ** (252.0 / len(df_nav)) - 1.0
    sharpe = (df_nav['nav'].pct_change().fillna(0.0).mean() / (df_nav['nav'].pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    
    cum_max = df_nav['nav'].cummax()
    max_dd = ((df_nav['nav'] - cum_max) / cum_max).min()

    sells = df_trades[df_trades['side'] == 'SELL'] if not df_trades.empty and 'side' in df_trades.columns else pd.DataFrame()
    win_rate = (sells['net_pnl'] > 0).mean() if not sells.empty else 0.0

    print("\n" + "="*75)
    print("      【机构级严谨时间结构管道与 10 级全诊断】测试报告")
    print("="*75)
    print("时间结构对齐:       feature_date (T-1) < trade_date (T)")
    print("盘中时间戳结构:     signal_time (09:45) < execution_time (10:00)")
    print("零缺省放行规则:     彻底删除所有 .fillna 补齐，缺失直接过滤！")
    print("-" * 75)
    print("【逐字段 PIT 覆盖率统计 (Per-Field PIT Coverage)】")
    print("  - 正股 T-1 收盘价 (stk_close_t1): {:.1f}% ({:,} 行)".format(stk_t1_cov, df_pit['stk_close_t1'].notnull().sum()))
    print("  - 转股价 (conv_price):            {:.1f}% ({:,} 行)".format(conv_px_cov, df_pit['conv_price'].notnull().sum()))
    print("  - T-1 转股价值 (conv_value_t1):   {:.1f}% ({:,} 行)".format(conv_val_cov, df_pit['conv_value_t1'].notnull().sum()))
    print("  - 发行规模 (curr_iss_amt):        {:.1f}% ({:,} 行)".format(iss_amt_cov, df_pit['curr_iss_amt'].notnull().sum()))
    print("  - 强赎状态 (is_redeemed):         {:.1f}% ({:,} 行)".format(redeem_cov, df_pit['is_redeemed'].notnull().sum()))
    print("  - T-1 选债合格率 (is_eligible):   {:.1f}% ({:,} 行)".format(eligible_ratio, df_pit['is_eligible_at_selection'].sum()))
    print("-" * 75)
    print("【10 级全诊断管道汇总统计 (日均)】")
    print("  - 每日原始债池数量:             {:.1f} 只".format(df_diag['raw_bonds'].mean()))
    print("  - PIT 元数据完整数量:           {:.1f} 只".format(df_diag['pit_complete'].mean()))
    print("  - T-1 选债合格数量 (Eligible):   {:.1f} 只".format(df_diag['eligible_bonds'].mean()))
    print("  - T 日目标池选出数量 (Top 10):  {:.1f} 只".format(df_diag['top_n_selected'].mean()))
    print("  - 盘中 15m 触发信号数量:        {:.1f} 只".format(df_diag['intraday_signals'].mean()))
    print("  - 生成时间结构订单总笔数:       {} 笔".format(len(df_orders)))
    print("  - 真实撮合成交笔数:             {} 笔 (买入 {} 笔, 卖出 {} 笔)".format(len(df_trades), len(df_trades[df_trades['side']=='BUY']) if not df_trades.empty else 0, len(sells)))
    print("-" * 75)
    print("【全新架构可信绩效表现 (2025.01 ~ 2026.07)】")
    print("  - 初始资金:                     1,000,000.00 元")
    print("  - 最终资金:                     {:,.2f} 元".format(df_nav['nav'].iloc[-1]))
    print("  - 累计净收益率:                 {:+.2f}%".format(total_ret * 100))
    print("  - 年化收益率 (Annualized Return):{:+.2f}%".format(ann_ret * 100))
    print("  - 夏普比率 (Sharpe Ratio):      {:.2f}".format(sharpe))
    print("  - 最大回撤 (Max Drawdown):      {:.2f}%".format(max_dd * 100))
    print("  - 平仓胜率 (Win Rate):              {:.1f}%".format(win_rate * 100))
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
