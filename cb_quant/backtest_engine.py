# -*- coding: utf-8 -*-

"""
可转债 T+0 分钟级事件驱动回测引擎 (包含动态止损与离场补位 - 高效向量化版)
Convertible Bond T+0 Minute Event-Driven Backtest Simulator (High-Performance Vectorized Edition)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBMinuteBacktestEngine:
    def __init__(self, initial_capital=1000000.0, top_n=8, rank_decay_buffer=5,
                 stop_loss_pct=0.005, single_slippage=0.001, commission=0.00005):
        """
        initial_capital: 初始资金 (100万)
        top_n: 目标最大持仓只数 N (8 只)
        rank_decay_buffer: 离场排名缓冲 (8 + 5 = 13 名)
        stop_loss_pct: 动态止损比例 (0.5%)
        single_slippage: 单边滑点 (0.1%)
        commission: 佣金 (万0.5)
        """
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.rank_decay_threshold = top_n + rank_decay_buffer
        self.stop_loss_pct = stop_loss_pct
        self.single_slippage = single_slippage
        self.commission = commission

    def run_backtest(self, df_scored):
        df = df_scored.copy()
        df['trade_time'] = df['trade_time'].astype(str)
        
        # 预先整理以 trade_time 为 key 的字典组
        time_groups = dict(tuple(df.groupby('trade_time')))
        timestamps = sorted(time_groups.keys())
        
        capital = self.initial_capital
        positions = {}
        trade_logs = []
        equity_curve = []
        
        slot_target_value = capital / self.top_n
        logger.info(f"开始分钟级回测，处理 {len(timestamps):,} 个 K 线时间节点...")

        for t in timestamps:
            df_t = time_groups[t]
            t_dict = df_t.set_index('ts_code').to_dict('index')
            
            # 1. 计算总市值与 NAV
            total_position_val = 0.0
            for code, pos in positions.items():
                if code in t_dict:
                    px = t_dict[code]['close']
                    total_position_val += pos['shares'] * px
                else:
                    total_position_val += pos['shares'] * pos['entry_price']

            nav = capital + total_position_val
            equity_curve.append({
                'trade_time': t, 
                'nav': nav, 
                'cash': capital, 
                'position_val': total_position_val, 
                'num_positions': len(positions)
            })
            slot_target_value = nav / self.top_n

            # 2. 止损检查与离场
            to_remove = []
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                entry_px = pos['entry_price']
                current_low = row['low']
                min30_low = row.get('low_min30m', 0.0)

                ret_from_entry = (current_low - entry_px) / entry_px
                break_min30 = (current_low <= min30_low) if (pd.notna(min30_low) and min30_low > 0) else False

                if ret_from_entry <= -self.stop_loss_pct or break_min30:
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

            # 3. 排名衰减卖出 (掉出 Top 13)
            to_decay_sell = []
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                rank = row.get('rank', 9999)
                if rank > self.rank_decay_threshold:
                    sell_px = row['close'] * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    entry_px = pos['entry_price']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    pnl_pct = (sell_px / entry_px) - 1.0
                    trade_logs.append({
                        'trade_time': t, 'ts_code': code, 'action': 'RANK_DECAY_SELL',
                        'price': sell_px, 'shares': shares, 'pnl': pnl, 'pnl_pct': pnl_pct
                    })
                    to_decay_sell.append(code)

            for code in to_decay_sell:
                del positions[code]

            # 4. 动态建仓与离场补位 (补满 8 只空位)
            open_slots = self.top_n - len(positions)
            if open_slots > 0 and capital > 10000.0:
                # 筛选当前不在持仓中且排名前 Top N 的目标
                candidates = [
                    (code, row['close'], row['rank']) 
                    for code, row in t_dict.items() 
                    if row.get('rank', 9999) <= self.top_n and code not in positions
                ]
                candidates.sort(key=lambda x: x[2]) # 按 rank 升序

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
                            'entry_time': t
                        }
                        trade_logs.append({
                            'trade_time': t, 'ts_code': code, 'action': 'BUY',
                            'price': buy_px, 'shares': shares, 'pnl': 0.0, 'pnl_pct': 0.0
                        })

        return pd.DataFrame(equity_curve), pd.DataFrame(trade_logs)
