# -*- coding: utf-8 -*-

"""
P0 级机构研究基础与回测可信度引擎 (严格下一根 K 线 t+1 开盘价成交与六项成本显式拆解)
P0 Institutional Research Foundation Engine (Strict Next-Bar Open Execution & Explicit 6-Cost Breakdown)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBP0PIPEngine:
    def __init__(self, initial_capital=1000000.0, top_n=5,
                 single_slippage=0.0005, single_impact=0.0005, commission=0.00005):
        """
        P0 显式拆解成本参数：
        - single_slippage: 0.05% 单边买卖价差
        - single_impact: 0.05% 单边冲击成本
        - commission: 0.005% (万0.5) 佣金与规费
        """
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.single_slippage = single_slippage
        self.single_impact = single_impact
        self.commission = commission
        self.total_friction_one_way = single_slippage + single_impact + commission

    def run_strict_next_bar_backtest(self, df_scored):
        """
        P0 核心规则：
        1. 必须在 bar t 结束时根据 score_15m 生成信号；
        2. 必须在下一个 K 线 bar t+1 的 Open 开盘价挂单成交 (绝对禁止同根 K 线按 Close 成交！);
        3. 导出六项成本拆解明细。
        """
        df = df_scored.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)
        
        # 构建 Next-Bar Open 字段 (t+1 开盘价)
        df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
        df['next_trade_time'] = df.groupby('ts_code')['trade_time'].shift(-1)
        
        time_groups = dict(tuple(df.groupby('trade_time')))
        timestamps = sorted(time_groups.keys())
        
        capital = self.initial_capital
        positions = {} # ts_code -> {'shares', 'entry_price', 'entry_open_price', 'entry_time', 'bars_held'}
        pending_orders = [] # [(ts_code, action, target_shares, signal_time)]
        
        trade_logs = []
        equity_curve = []
        slot_target_value = capital / self.top_n

        logger.info(f"启动 P0 机构级严格 Next-Bar ($t+1$) 开盘价成交仿真...")

        for b_idx, t in enumerate(timestamps):
            df_t = time_groups[t]
            t_dict = df_t.set_index('ts_code').to_dict('index')
            time_str = str(t)[11:16]
            
            # --- 步骤 A：执行在上一根 K 线 (t-1) 产生的待平仓/开仓委托 (按当前 bar t 的 Open 开盘价成交) ---
            new_pending_orders = []
            for p_order in pending_orders:
                code = p_order['ts_code']
                action = p_order['action']
                if code not in t_dict:
                    continue
                
                execution_open_px = t_dict[code]['open'] # 严格按 $t+1$ Open 开盘价成交！
                
                if action == 'BUY':
                    buy_gross_px = execution_open_px
                    buy_net_px = buy_gross_px * (1.0 + self.total_friction_one_way)
                    
                    target_alloc = min(slot_target_value, capital * 0.95)
                    shares = target_alloc / buy_net_px
                    
                    if shares >= 10 and capital >= shares * buy_net_px:
                        cost = shares * buy_net_px
                        capital -= cost
                        positions[code] = {
                            'shares': shares,
                            'entry_gross_price': buy_gross_px,
                            'entry_net_price': buy_net_px,
                            'entry_time': str(t),
                            'bars_held': 0
                        }
                        trade_logs.append({
                            'trade_time': str(t), 'ts_code': code, 'action': 'BUY_NEXT_OPEN',
                            'gross_price': buy_gross_px, 'net_price': buy_net_px, 'shares': shares,
                            'gross_pnl': 0.0, 'net_pnl': 0.0, 'slippage_cost': shares * buy_gross_px * self.single_slippage,
                            'impact_cost': shares * buy_gross_px * self.single_impact,
                            'commission_cost': shares * buy_gross_px * self.commission
                        })
                        
                elif action == 'SELL':
                    if code in positions:
                        pos = positions[code]
                        shares = pos['shares']
                        sell_gross_px = execution_open_px
                        sell_net_px = sell_gross_px * (1.0 - self.total_friction_one_way)
                        
                        revenue = shares * sell_net_px
                        capital += revenue
                        
                        gross_pnl = shares * (sell_gross_px - pos['entry_gross_price'])
                        net_pnl = revenue - (shares * pos['entry_net_price'])
                        
                        trade_logs.append({
                            'trade_time': str(t), 'ts_code': code, 'action': p_order['reason'],
                            'gross_price': sell_gross_px, 'net_price': sell_net_px, 'shares': shares,
                            'gross_pnl': gross_pnl, 'net_pnl': net_pnl,
                            'slippage_cost': shares * sell_gross_px * self.single_slippage,
                            'impact_cost': shares * sell_gross_px * self.single_slippage,
                            'commission_cost': shares * sell_gross_px * self.commission
                        })
                        del positions[code]

            pending_orders = [] # 清空已执行委托

            # --- 步骤 B：评估当前时刻 NAV ---
            total_pos_val = 0.0
            for code, pos in positions.items():
                pos['bars_held'] += 1
                if code in t_dict:
                    total_pos_val += pos['shares'] * t_dict[code]['close']
                else:
                    total_pos_val += pos['shares'] * pos['entry_gross_price']

            nav = capital + total_pos_val
            equity_curve.append({
                'trade_time': str(t),
                'nav': nav,
                'cash': capital,
                'position_val': total_pos_val,
                'num_positions': len(positions)
            })
            slot_target_value = nav / self.top_n

            # --- 步骤 C：在当前 bar t 结束时生成信号，下达待在 $t+1$ Open 执行的委托 ---
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                rank_15m = row.get('rank_15m', 999)
                curr_ret = (row['close'] - pos['entry_gross_price']) / pos['entry_gross_price']

                # 离场信号条件
                if curr_ret <= -0.010: # -1.0% 硬止损
                    pending_orders.append({'ts_code': code, 'action': 'SELL', 'reason': 'P0_STOP_LOSS_-1%'})
                elif rank_15m > 15: # 排名掉出 Top 15 软换仓
                    pending_orders.append({'ts_code': code, 'action': 'SELL', 'reason': 'P0_RANK_EXIT'})
                elif pos['bars_held'] >= 6 or time_str >= '14:30': # 90m 或 14:30 离场
                    pending_orders.append({'ts_code': code, 'action': 'SELL', 'reason': 'P0_TIME_EXIT'})

            # 开仓买入信号
            open_slots = self.top_n - len(positions) - len([p for p in pending_orders if p['action'] == 'BUY'])
            if open_slots > 0 and capital > 10000.0:
                candidates = []
                for code, row in t_dict.items():
                    rank_15m = row.get('rank_15m', 999)
                    if rank_15m <= self.top_n and code not in positions and code not in [p['ts_code'] for p in pending_orders]:
                        candidates.append((code, rank_15m))

                candidates.sort(key=lambda x: x[1])
                for code, rank_val in candidates[:open_slots]:
                    pending_orders.append({'ts_code': code, 'action': 'BUY', 'reason': 'P0_BUY_SIGNAL'})

        return pd.DataFrame(equity_curve), pd.DataFrame(trade_logs)
