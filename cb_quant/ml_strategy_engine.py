# -*- coding: utf-8 -*-

"""
升维可转债 T+0 事件驱动回测引擎 (具备三层主动止盈体系：移动保本 + 动态跟踪止盈 + 目标时间止盈)
Upgraded CB T+0 Event-Driven Backtest Simulator with Three-Layer Active Take-Profit
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBMLStrategyEngine:
    def __init__(self, initial_capital=1000000.0, top_n=8, exit_rank_k=12,
                 atr_multiplier=1.5, max_hold_bars=24, min_amp_threshold=0.008,
                 cooldown_bars=6, single_slippage=0.001, commission=0.00005,
                 break_even_trigger=0.015, trailing_trigger=0.025, trailing_pct=0.012):
        """
        break_even_trigger: 浮盈达到 1.5% 触发移动保本
        trailing_trigger: 浮盈达到 2.5% 触发峰值跟踪止盈
        trailing_pct: 峰值回撤 1.2% 锁定利润
        """
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.exit_rank_k = exit_rank_k
        self.atr_multiplier = atr_multiplier
        self.max_hold_bars = max_hold_bars
        self.min_amp_threshold = min_amp_threshold
        self.cooldown_bars = cooldown_bars
        self.single_slippage = single_slippage
        self.commission = commission
        self.break_even_trigger = break_even_trigger
        self.trailing_trigger = trailing_trigger
        self.trailing_pct = trailing_pct

    def run_backtest(self, df_predicted):
        df = df_predicted.copy()
        df['trade_time'] = df['trade_time'].astype(str)
        
        time_groups = dict(tuple(df.groupby('trade_time')))
        timestamps = sorted(time_groups.keys())
        
        capital = self.initial_capital
        positions = {} # ts_code -> {'shares', 'entry_price', 'entry_time', 'bars_held', 'stop_dist', 'peak_price', 'is_breakeven', 'is_trailing'}
        cooldown_dict = {}
        
        trade_logs = []
        equity_curve = []
        slot_target_value = capital / self.top_n

        logger.info(f"启动三层止盈 T+0 极速回测 (持仓 N={self.top_n}, 保本={self.break_even_trigger*100}%, 跟踪={self.trailing_trigger*100}%)...")

        for b_idx, t in enumerate(timestamps):
            df_t = time_groups[t]
            t_dict = df_t.set_index('ts_code').to_dict('index')
            
            # 1. 评估 NAV
            total_pos_val = 0.0
            for code, pos in positions.items():
                pos['bars_held'] += 1
                if code in t_dict:
                    px = t_dict[code]['close']
                    total_pos_val += pos['shares'] * px
                    # 更新持仓最高价
                    pos['peak_price'] = max(pos['peak_price'], t_dict[code]['high'])
                else:
                    total_pos_val += pos['shares'] * pos['entry_price']

            nav = capital + total_pos_val
            equity_curve.append({
                'trade_time': t,
                'nav': nav,
                'cash': capital,
                'position_val': total_pos_val,
                'num_positions': len(positions)
            })
            slot_target_value = nav / self.top_n

            # 2. 三层主动止盈与动态 ATR 止损检查
            exited_codes = []
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                entry_px = pos['entry_price']
                current_high = row['high']
                current_low = row['low']
                current_close = row['close']
                peak_px = pos['peak_price']

                # 计算峰值收益率与浮动收益率
                peak_ret = (peak_px - entry_px) / entry_px
                curr_ret = (current_close - entry_px) / entry_px

                # --- 第一层：保本止损 (移动保本) ---
                if peak_ret >= self.break_even_trigger and not pos['is_breakeven']:
                    pos['is_breakeven'] = True
                    # 将止损价提升至成本价 + 0.1% (保本覆盖交易成本)
                    pos['stop_price'] = max(pos.get('stop_price', 0.0), entry_px * 1.001)

                # --- 第二层：动态跟踪止盈 (Trailing Stop) ---
                if peak_ret >= self.trailing_trigger:
                    pos['is_trailing'] = True
                    # 止盈价设定为 峰值回撤 1.2%
                    trailing_stop_px = peak_px * (1.0 - self.trailing_pct)
                    pos['stop_price'] = max(pos.get('stop_price', 0.0), trailing_stop_px)

                stop_px = pos.get('stop_price', entry_px - pos['stop_dist'])

                # 触及止损价/止盈价平仓
                if current_low <= stop_px:
                    sell_px = current_low * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    pnl_pct = (sell_px / entry_px) - 1.0
                    action_type = 'TRAILING_TAKE_PROFIT' if pos['is_trailing'] else ('BREAK_EVEN_EXIT' if pos['is_breakeven'] else 'ATR_STOP_LOSS')
                    
                    trade_logs.append({
                        'trade_time': t, 'ts_code': code, 'action': action_type,
                        'price': sell_px, 'shares': shares, 'pnl': pnl, 'pnl_pct': pnl_pct
                    })
                    exited_codes.append(code)
                    cooldown_dict[code] = b_idx + self.cooldown_bars
                    continue

                # --- 第三层：时间目标止盈 (持仓 >= 60m 且 浮盈 >= +1.0% 主动落袋为安) ---
                if pos['bars_held'] >= 12 and curr_ret >= 0.010:
                    sell_px = current_close * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    pnl_pct = (sell_px / entry_px) - 1.0
                    trade_logs.append({
                        'trade_time': t, 'ts_code': code, 'action': 'TARGET_TIME_TAKE_PROFIT',
                        'price': sell_px, 'shares': shares, 'pnl': pnl, 'pnl_pct': pnl_pct
                    })
                    exited_codes.append(code)
                    cooldown_dict[code] = b_idx + self.cooldown_bars
                    continue

                # 4. 时间平仓 (持仓 > 120 分钟未盈利平仓)
                if pos['bars_held'] >= self.max_hold_bars and curr_ret <= 0:
                    sell_px = current_close * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    pnl_pct = (sell_px / entry_px) - 1.0
                    trade_logs.append({
                        'trade_time': t, 'ts_code': code, 'action': 'TIME_EXIT_SELL',
                        'price': sell_px, 'shares': shares, 'pnl': pnl, 'pnl_pct': pnl_pct
                    })
                    exited_codes.append(code)
                    cooldown_dict[code] = b_idx + self.cooldown_bars
                    continue

                # 5. 软止损检查 (排名掉出 Top 12 离场)
                rank = row.get('rank', 9999)
                if rank > self.exit_rank_k:
                    sell_px = current_close * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    pnl_pct = (sell_px / entry_px) - 1.0
                    trade_logs.append({
                        'trade_time': t, 'ts_code': code, 'action': 'RANK_EXIT_SELL',
                        'price': sell_px, 'shares': shares, 'pnl': pnl, 'pnl_pct': pnl_pct
                    })
                    exited_codes.append(code)
                    cooldown_dict[code] = b_idx + self.cooldown_bars

            for code in exited_codes:
                if code in positions:
                    del positions[code]

            # 6. 波动率过滤 + 置信度即时补位买入
            open_slots = self.top_n - len(positions)
            if open_slots > 0 and capital > 10000.0:
                candidates = []
                for code, row in t_dict.items():
                    rank = row.get('rank', 9999)
                    amp = row.get('amp_15m', 0.0)
                    
                    if rank <= self.exit_rank_k and amp >= self.min_amp_threshold and code not in positions and cooldown_dict.get(code, 0) <= b_idx:
                        candidates.append((code, row['close'], rank, row.get('atr_14', 1.0)))

                candidates.sort(key=lambda x: x[2])

                for code, close_px, rank, atr_val in candidates:
                    if len(positions) >= self.top_n or capital < 10000.0:
                        break
                    
                    buy_px = close_px * (1.0 + self.single_slippage)
                    target_alloc = min(slot_target_value, capital * 0.95)
                    shares = target_alloc / (buy_px * (1.0 + self.commission))
                    
                    if shares >= 10:
                        cost = shares * buy_px * (1.0 + self.commission)
                        capital -= cost
                        stop_dist = max(buy_px * 0.015, self.atr_multiplier * atr_val)
                        
                        positions[code] = {
                            'shares': shares,
                            'entry_price': buy_px,
                            'entry_time': t,
                            'bars_held': 0,
                            'stop_dist': stop_dist,
                            'peak_price': buy_px,
                            'is_breakeven': False,
                            'is_trailing': False,
                            'stop_price': buy_px - stop_dist
                        }
                        trade_logs.append({
                            'trade_time': t, 'ts_code': code, 'action': 'BUY',
                            'price': buy_px, 'shares': shares, 'pnl': 0.0, 'pnl_pct': 0.0
                        })

        return pd.DataFrame(equity_curve), pd.DataFrame(trade_logs)
