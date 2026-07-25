# -*- coding: utf-8 -*-

"""
15分钟日内 T+0 零隔夜策略引擎 (结合双低风控围栏、15m均值回归反转与早盘高波集中)
Intraday 15-Minute T+0 Zero-Overnight Strategy Engine
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBIntraday15mEngine:
    def __init__(self, initial_capital=1000000.0, top_n=5,
                 take_profit=0.015, stop_loss=0.008, max_hold_bars=4,
                 single_slippage=0.0005, commission=0.00005):
        """
        initial_capital: 初始资金 100万
        top_n: 集中持仓 Top 3~5 只
        take_profit: 硬止盈 +1.5% (见好就收)
        stop_loss: 硬止损 -0.8% (小止损严控)
        max_hold_bars: 最大持仓 4 根 K线 (60分钟，超时强平)
        single_slippage: 0.05% 单边贴近真实盘口滑点
        commission: 万0.5 佣金
        """
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold_bars = max_hold_bars
        self.single_slippage = single_slippage
        self.commission = commission

    def run_backtest(self, df_15m):
        df = df_15m.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df['date'] = df['trade_time'].dt.date
        df['time_str'] = df['trade_time'].dt.strftime('%H:%M')
        
        # 逐日内节点分组
        time_groups = dict(tuple(df.groupby('trade_time')))
        timestamps = sorted(time_groups.keys())
        
        capital = self.initial_capital
        positions = {}
        trade_logs = []
        equity_curve = []
        slot_target_value = capital / self.top_n

        logger.info(f"启动 15分钟日内 T+0 零隔夜回测 (持仓 N={self.top_n}, 止盈=+{self.take_profit*100}%, 止损=-{self.stop_loss*100}%, 零隔夜)...")

        for b_idx, t in enumerate(timestamps):
            df_t = time_groups[t]
            t_dict = df_t.set_index('ts_code').to_dict('index')
            time_str = df_t['time_str'].iloc[0]
            
            # 1. 评估 NAV
            total_pos_val = 0.0
            for code, pos in positions.items():
                pos['bars_held'] += 1
                if code in t_dict:
                    total_pos_val += pos['shares'] * t_dict[code]['close']
                else:
                    total_pos_val += pos['shares'] * pos['entry_price']

            nav = capital + total_pos_val
            equity_curve.append({
                'trade_time': str(t),
                'nav': nav,
                'cash': capital,
                'position_val': total_pos_val,
                'num_positions': len(positions)
            })
            slot_target_value = nav / self.top_n

            # 2. 日内硬性离场与平仓规则
            exited_codes = []
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                entry_px = pos['entry_price']
                current_high = row['high']
                current_low = row['low']
                current_close = row['close']
                
                high_ret = (current_high - entry_px) / entry_px
                low_ret = (current_low - entry_px) / entry_px
                curr_ret = (current_close - entry_px) / entry_px

                # A. 硬止盈 (+0.7% 快速抢反弹平仓)
                if high_ret >= self.take_profit:
                    sell_px = entry_px * (1.0 + self.take_profit) * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    trade_logs.append({'trade_time': str(t), 'ts_code': code, 'action': 'INTRA_TAKE_PROFIT', 'price': sell_px, 'shares': shares, 'pnl': pnl})
                    exited_codes.append(code)
                    continue

                # B. 硬止损 (-0.5% 快速剪裁)
                if low_ret <= -self.stop_loss:
                    sell_px = entry_px * (1.0 - self.stop_loss) * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    trade_logs.append({'trade_time': str(t), 'ts_code': code, 'action': 'INTRA_STOP_LOSS', 'price': sell_px, 'shares': shares, 'pnl': pnl})
                    exited_codes.append(code)
                    continue

                # C. 14:45 强制尾盘清仓 (零隔夜) 或 超时 60 分钟平仓
                if time_str >= '14:45' or pos['bars_held'] >= self.max_hold_bars:
                    sell_px = current_close * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    action = 'FORCE_CLOSE_1445' if time_str >= '14:45' else 'TIME_EXIT_60M'
                    trade_logs.append({'trade_time': str(t), 'ts_code': code, 'action': action, 'price': sell_px, 'shares': shares, 'pnl': pnl})
                    exited_codes.append(code)

            for code in exited_codes:
                if code in positions:
                    del positions[code]

            # 3. 集中早盘开仓 (09:45 ~ 11:15 抓住高波窗口)
            if '09:45' <= time_str <= '11:15':
                open_slots = self.top_n - len(positions)
                if open_slots > 0 and capital > 10000.0:
                    candidates = []
                    for code, row in t_dict.items():
                        # 融合规则：必须在双低围栏内 + 15m 价格微幅回踩反转 (Mean-Reversion)
                        if row.get('in_double_low_pool', False) and code not in positions:
                            ret_15m = row.get('ret_15m', 0.0)
                            # 避开脉冲追高，选择双低池内微幅回调的优质标的
                            if -0.012 <= ret_15m <= -0.002:
                                candidates.append((code, row['close'], row.get('double_low', 999.0)))

                    candidates.sort(key=lambda x: x[2]) # 按双低低值排序

                    for code, close_px, d_low in candidates:
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
                                'entry_time': str(t),
                                'bars_held': 0
                            }
                            trade_logs.append({'trade_time': str(t), 'ts_code': code, 'action': 'INTRA_BUY', 'price': buy_px, 'shares': shares, 'pnl': 0.0})

        return pd.DataFrame(equity_curve), pd.DataFrame(trade_logs)
