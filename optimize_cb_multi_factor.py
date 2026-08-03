# -*- coding: utf-8 -*-

"""
可转债 5 分钟线多因子策略参数优化与稳健性提升脚本
Convertible Bond 5-Min Multi-Factor Strategy Parameter Optimization Engine
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader
from cb_quant.factor_engine import CBFactorEngine
from cb_quant.scoring_engine import CBScoringEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class RobustCBBacktestEngine:
    def __init__(self, initial_capital=1000000.0, top_n=5, min_hold_bars=6,
                 stop_loss_pct=0.015, single_slippage=0.001, commission=0.00005):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.min_hold_bars = min_hold_bars
        self.stop_loss_pct = stop_loss_pct
        self.single_slippage = single_slippage
        self.commission = commission

    def run_backtest(self, df_scored):
        df = df_scored.copy()
        df['trade_time'] = df['trade_time'].astype(str)
        time_groups = dict(tuple(df.groupby('trade_time')))
        timestamps = sorted(time_groups.keys())
        
        capital = self.initial_capital
        positions = {} # ts_code -> {'shares': float, 'entry_price': float, 'entry_time': str, 'bars_held': int}
        trade_logs = []
        equity_curve = []
        
        slot_target_value = capital / self.top_n

        for t_idx, t in enumerate(timestamps):
            df_t = time_groups[t]
            t_dict = df_t.set_index('ts_code').to_dict('index')
            
            # 1. 评估 NAV
            total_position_val = 0.0
            for code, pos in positions.items():
                pos['bars_held'] += 1
                if code in t_dict:
                    px = t_dict[code]['close']
                    total_position_val += pos['shares'] * px
                else:
                    total_position_val += pos['shares'] * pos['entry_price']

            nav = capital + total_position_val
            equity_curve.append({'trade_time': t, 'nav': nav, 'cash': capital, 'position_val': total_position_val, 'num_positions': len(positions)})
            slot_target_value = nav / self.top_n

            # 2. 止损判断 (提升为 1.5% - 2.0% 宽止损，且需达到最小持仓时间)
            to_remove = []
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                entry_px = pos['entry_price']
                current_low = row['low']

                ret_from_entry = (current_low - entry_px) / entry_px

                if ret_from_entry <= -self.stop_loss_pct:
                    sell_px = current_low * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    pnl_pct = (sell_px / entry_px) - 1.0
                    trade_logs.append({
                        'trade_time': t, 'ts_code': code, 'action': 'STOP_LOSS',
                        'price': sell_px, 'shares': shares, 'pnl': pnl, 'pnl_pct': pnl_pct
                    })
                    to_remove.append(code)

            for code in to_remove:
                del positions[code]

            # 3. 轮换卖出 (须满足最小持仓 K 线数，避免高频过热度损耗)
            to_decay_sell = []
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                rank = row.get('rank', 9999)
                
                # 满最小持仓时间后，若掉出 Top 10 卖出
                if pos['bars_held'] >= self.min_hold_bars and rank > (self.top_n + 5):
                    sell_px = row['close'] * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    entry_px = pos['entry_price']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    pnl_pct = (sell_px / entry_px) - 1.0
                    trade_logs.append({
                        'trade_time': t, 'ts_code': code, 'action': 'RANK_ROTATE_SELL',
                        'price': sell_px, 'shares': shares, 'pnl': pnl, 'pnl_pct': pnl_pct
                    })
                    to_decay_sell.append(code)

            for code in to_decay_sell:
                del positions[code]

            # 4. 选优补位买入
            open_slots = self.top_n - len(positions)
            if open_slots > 0 and capital > 10000.0:
                candidates = [
                    (code, row['close'], row['rank']) 
                    for code, row in t_dict.items() 
                    if row.get('rank', 9999) <= self.top_n and code not in positions
                ]
                candidates.sort(key=lambda x: x[2])

                for code, close_px, rank in candidates:
                    if len(positions) >= self.top_n or capital < 10000.0:
                        break
                    
                    buy_px = close_px * (1.0 + self.single_slippage)
                    target_alloc = min(slot_target_value, capital * 0.95)
                    shares = target_alloc / (buy_px * (1.0 + self.commission))
                    
                    if shares >= 10:
                        cost = shares * buy_px * (1.0 + self.commission)
                        capital -= cost
                        positions[code] = {
                            'shares': shares,
                            'entry_price': buy_px,
                            'entry_time': t,
                            'bars_held': 0
                        }
                        trade_logs.append({
                            'trade_time': t, 'ts_code': code, 'action': 'BUY',
                            'price': buy_px, 'shares': shares, 'pnl': 0.0, 'pnl_pct': 0.0
                        })

        return pd.DataFrame(equity_curve), pd.DataFrame(trade_logs)

def evaluate(df_eq, df_tr):
    if df_eq.empty:
        return 0.0, 0.0, 0.0, 0
    ret = (df_eq['nav'].iloc[-1] - 1000000.0) / 1000000.0
    cummax = df_eq['nav'].cummax()
    mdd = ((df_eq['nav'] - cummax) / cummax).min()
    trades = len(df_tr[df_tr['action'] != 'BUY'])
    win = len(df_tr[df_tr['pnl'] > 0]) / max(trades, 1)
    return ret, mdd, win, trades

def main():
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=150)
    df_factors = CBFactorEngine.compute_factors(df_panel)
    
    scoring_engine = CBScoringEngine(top_n=5, min_amount=500000.0)
    df_scored = scoring_engine.compute_cross_sectional_scores(df_factors)
    
    print("=== 开始参数网格优化 (寻找稳健超额收益组合) ===")
    for hold_bars in [6, 12, 24]: # 30min, 60min, 120min
        for stop_pct in [0.01, 0.015, 0.02]: # 1.0%, 1.5%, 2.0% 宽止损
            engine = RobustCBBacktestEngine(
                initial_capital=1000000.0,
                top_n=5,
                min_hold_bars=hold_bars,
                stop_loss_pct=stop_pct,
                single_slippage=0.001,
                commission=0.00005
            )
            df_eq, df_tr = engine.run_backtest(df_scored)
            ret, mdd, win, trades = evaluate(df_eq, df_tr)
            print(f"HoldBars: {hold_bars:2d} ({(hold_bars*5)}m) | StopLoss: {stop_pct*100:3.1f}% | 收益率: {ret*100:6.2f}% | 最大回撤: {mdd*100:6.2f}% | 胜率: {win*100:5.1f}% | 总交易: {trades} 次")

if __name__ == "__main__":
    main()
