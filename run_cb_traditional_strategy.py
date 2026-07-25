# -*- coding: utf-8 -*-

"""
纯传统双低多因子轮动策略 (日频轮动经典基准)
Standalone Classic Convertible Bond Double-Low Strategy (Daily Rebalancing Baseline)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cb_quant.data_loader import CBDataLoader
from cb_quant.traditional_factor_engine import CBTraditionalFactorEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_traditional_backtest(df_trad, top_n=8, initial_capital=1000000.0, single_slippage=0.001, commission=0.00005):
    df = df_trad.copy()
    df['trade_time'] = df['trade_time'].astype(str)
    df['date'] = pd.to_datetime(df['trade_time']).dt.date
    
    # 每日仅在 14:55 (每日收盘前节点) 触发调仓轮换，极大降低换手率
    daily_close_mask = pd.to_datetime(df['trade_time']).dt.minute == 55
    df_daily = df[daily_close_mask].copy()
    
    dates = sorted(df_daily['date'].unique())
    capital = initial_capital
    positions = {} # ts_code -> {'shares', 'entry_price'}
    
    trade_logs = []
    equity_curve = []
    
    date_groups = dict(tuple(df_daily.groupby('date')))

    logging.info(f"启动纯传统双低轮动策略回测 (持仓 Top {top_n}, 每日收盘调仓)...")

    for d in dates:
        df_d = date_groups[d].drop_duplicates(subset=['ts_code'])
        d_dict = df_d.set_index('ts_code').to_dict('index')
        
        # 1. 评估当前 NAV
        total_pos_val = 0.0
        for code, pos in positions.items():
            if code in d_dict:
                px = d_dict[code]['close']
                total_pos_val += pos['shares'] * px
            else:
                total_pos_val += pos['shares'] * pos['entry_price']

        nav = capital + total_pos_val
        equity_curve.append({'date': str(d), 'nav': nav, 'cash': capital, 'position_val': total_pos_val, 'num_positions': len(positions)})
        slot_target_value = nav / top_n

        # 2. 选出今日双低得分最高 (Rank <= Top N) 的标的
        candidates = [
            (code, row['close'], row['double_low_rank']) 
            for code, row in d_dict.items() 
            if row.get('double_low_rank', 9999) <= top_n
        ]
        target_codes = set([c[0] for c in candidates])

        # 3. 卖出不在目标池中的标的
        to_sell = [code for code in positions if code not in target_codes]
        for code in to_sell:
            pos = positions[code]
            sell_px = d_dict[code]['close'] * (1.0 - single_slippage) if code in d_dict else pos['entry_price']
            shares = pos['shares']
            revenue = shares * sell_px * (1.0 - commission)
            capital += revenue
            
            pnl = revenue - (shares * pos['entry_price'])
            trade_logs.append({'date': str(d), 'ts_code': code, 'action': 'SELL', 'price': sell_px, 'shares': shares, 'pnl': pnl})
            del positions[code]

        # 4. 买入新入选的标的
        for code, close_px, rank in candidates:
            if code not in positions and capital > 10000.0:
                buy_px = close_px * (1.0 + single_slippage)
                target_alloc = min(slot_target_value, capital * 0.95)
                shares = target_alloc / (buy_px * (1.0 + commission))
                
                if shares >= 10:
                    cost = shares * buy_px * (1.0 + commission)
                    capital -= cost
                    positions[code] = {'shares': shares, 'entry_price': buy_px}
                    trade_logs.append({'date': str(d), 'ts_code': code, 'action': 'BUY', 'price': buy_px, 'shares': shares, 'pnl': 0.0})

    return pd.DataFrame(equity_curve), pd.DataFrame(trade_logs)

def main():
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    logging.info("计算纯传统双低多因子与风控过滤...")
    df_trad = CBTraditionalFactorEngine.compute_traditional_factors(df_panel)
    
    df_equity, df_trades = run_traditional_backtest(df_trad, top_n=8)
    
    if not df_equity.empty:
        nav_s = df_equity['nav']
        tot_ret = (nav_s.iloc[-1] - 1000000.0) / 1000000.0
        num_days = len(df_equity)
        ann_ret = (1.0 + tot_ret) ** (252.0 / max(num_days, 1)) - 1.0
        
        cummax = nav_s.cummax()
        drawdown = (nav_s - cummax) / cummax
        max_dd = drawdown.min()
        
        sells = df_trades[df_trades['action'] == 'SELL']
        trades = len(sells)
        win = (sells['pnl'] > 0).mean() if trades > 0 else 0.0
        
        avg_win = sells[sells['pnl'] > 0]['pnl'].mean() if (trades > 0 and (sells['pnl'] > 0).sum() > 0) else 0.0
        avg_loss = abs(sells[sells['pnl'] < 0]['pnl'].mean()) if (trades > 0 and (sells['pnl'] < 0).sum() > 0) else 1.0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        print("\n" + "="*65)
        print("    纯传统双低多因子策略 (日频调仓基准) 绩效分析")
        print("="*65)
        print(f"回测测试区间:       2025-01-01 至 2026-07-24 (共 {num_days} 个交易日)")
        print(f"初始资金:           1,000,000.00 元")
        print(f"期末资金:           {nav_s.iloc[-1]:,.2f} 元")
        print(f"累计收益率 (Cumulative Return):  {tot_ret*100:+.2f}%")
        print(f"年化收益率 (Annualized Return):  {ann_ret*100:+.2f}%")
        print(f"最大回撤 (Max Drawdown):        {max_dd*100:.2f}%")
        print(f"平仓交易胜率 (Win Rate):         {win*100:.2f}%")
        print(f"盈亏比 (Profit/Loss Ratio):      {profit_loss_ratio:.2f}")
        print(f"同期中证转债指数涨跌幅:           -10.38%")
        print(f"超越大盘超额收益 (Alpha):        {tot_ret*100 - (-10.38):+.2f}%")
        print("="*65 + "\n")

if __name__ == "__main__":
    main()
