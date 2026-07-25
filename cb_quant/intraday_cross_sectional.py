# -*- coding: utf-8 -*-

"""
15分钟截面打分轮动与正股驱动滞后补涨引擎
15-Minute Cross-Sectional Ranking & Stock-Lag Driving Engine
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBIntradayCrossSectionalEngine:
    @staticmethod
    def filter_universe(df_panel, max_price=180.0, min_scale=2.0):
        """
        1. 建立交易池过滤：
        - 排除价格 > 180 元高估值妖债
        - 排除剩余规模 < 2.0 亿微型债
        - 排除已强赎转债
        """
        from cb_quant.traditional_factor_engine import CBTraditionalFactorEngine
        df = CBTraditionalFactorEngine.compute_traditional_factors(df_panel)
        
        # 兼容性备用：若缺少 curr_iss_amt 则填充默认值 5.0 亿
        if 'curr_iss_amt' not in df.columns:
            df['curr_iss_amt'] = 5.0
        if 'is_redeemed' not in df.columns:
            df['is_redeemed'] = False
            
        mask = (
            (df['close'] <= max_price) &
            (df['curr_iss_amt'] >= min_scale) &
            (df['is_redeemed'] == False)
        )
        return df[mask].copy()

    @staticmethod
    def compute_cross_sectional_scores(df_15m):
        """
        2. 15分钟截面多因子打分公式：
        Score = 0.35 * R15 + 0.30 * R60 + 0.20 * VolumeShock + 0.15 * CloseLocation - 0.25 * Volatility
        每个时间截面做去极值 (3sigma) 与 Z-Score 标准化！
        """
        df = df_15m.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)
        
        grouped = df.groupby('ts_code')
        
        # A. 基础因子计算
        df['close_lag1'] = grouped['close'].shift(1)
        df['close_lag4'] = grouped['close'].shift(4)
        
        df['ret_15m'] = (df['close'] - df['close_lag1']) / (df['close_lag1'] + 1e-8)
        df['ret_60m'] = (df['close'] - df['close_lag4']) / (df['close_lag4'] + 1e-8)
        
        # VolumeShock: 当前 15m 成交额 / 过去 20天同时间段均值
        df['vol_ma20'] = grouped['amount'].rolling(window=80, min_periods=10).mean().reset_index(level=0, drop=True)
        df['volume_shock'] = df['amount'] / (df['vol_ma20'] + 1e-5)
        
        # CloseLocation: (C - L) / (H - L)
        df['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Volatility: 近期 15m 波动率惩罚
        df['atr_15m'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        df['volatility_penalty'] = grouped['atr_15m'].rolling(window=8, min_periods=3).mean().reset_index(level=0, drop=True)

        # B. 逐 15m 时间截面去极值与 Z-Score 矢量化高速计算
        logger.info("进行 15分钟截面去极值、Z-Score 矢量化标准化与综合打分...")
        factors = ['ret_15m', 'ret_60m', 'volume_shock', 'close_location', 'volatility_penalty']
        
        for f in factors:
            mean = df.groupby('trade_time')[f].transform('mean')
            std = df.groupby('trade_time')[f].transform('std') + 1e-8
            
            # 去极值 3 sigma
            v_clip = df[f].clip(lower=mean - 3 * std, upper=mean + 3 * std)
            c_mean = df.groupby('trade_time')[f].transform('mean')
            c_std = df.groupby('trade_time')[f].transform('std') + 1e-8
            df[f + '_z'] = (v_clip - c_mean) / c_std

        # 综合合成得分
        df['score_15m'] = (
            0.35 * df['ret_15m_z'] +
            0.30 * df['ret_60m_z'] +
            0.20 * df['volume_shock_z'] +
            0.15 * df['close_location_z'] -
            0.25 * df['volatility_penalty_z']
        )
        
        df['rank_15m'] = df.groupby('trade_time')['score_15m'].rank(ascending=False, method='min')
        df['total_count'] = df.groupby('trade_time')['score_15m'].transform('count')
        df['rank_pct'] = df['rank_15m'] / (df['total_count'] + 1e-8)
        
        return df

class CBIntradayCrossSectionalSimulator:
    def __init__(self, initial_capital=1000000.0, top_n=8, exit_rank_pct=0.20,
                 stop_loss=-0.010, max_surge_cap=0.035,
                 single_slippage=0.0005, commission=0.00005):
        """
        MVP 交易规则：
        - 09:45 后开始交易，避免开盘噪声
        - 每 15m 重新计算截面排名，买入前 Top 5~8 名
        - 至少持有 30分钟 (2 根 K线)，最多持有 90分钟 (6 根 K线)
        - 跌出前 20% (exit_rank_pct) 则离场
        - 单债止损 -1.0%
        - 暴涨过滤：单根 15m 涨幅 > 3.5% 禁止追高
        - 14:45 强制平仓清仓 (零隔夜)
        """
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.exit_rank_pct = exit_rank_pct
        self.stop_loss = stop_loss
        self.max_surge_cap = max_surge_cap
        self.single_slippage = single_slippage
        self.commission = commission

    def run_backtest(self, df_scored):
        df = df_scored.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df['date'] = df['trade_time'].dt.date
        df['time_str'] = df['trade_time'].dt.strftime('%H:%M')
        
        time_groups = dict(tuple(df.groupby('trade_time')))
        timestamps = sorted(time_groups.keys())
        
        capital = self.initial_capital
        positions = {}
        trade_logs = []
        equity_curve = []
        slot_target_value = capital / self.top_n

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

            # 2. 持仓管理与离场检查
            exited_codes = []
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                entry_px = pos['entry_price']
                current_low = row['low']
                current_close = row['close']
                
                curr_ret = (current_close - entry_px) / entry_px
                low_ret = (current_low - entry_px) / entry_px
                rank_pct = row.get('rank_pct', 1.0)

                # A. 单债硬止损 (-1.0%)
                if low_ret <= self.stop_loss:
                    sell_px = entry_px * (1.0 + self.stop_loss) * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    trade_logs.append({'trade_time': str(t), 'ts_code': code, 'action': 'STOP_LOSS_-1%', 'price': sell_px, 'shares': shares, 'pnl': pnl})
                    exited_codes.append(code)
                    continue

                # B. 持有满 30分钟后：若跌出截面前 20% (rank_pct > 0.20) 换仓平仓
                if pos['bars_held'] >= 2 and rank_pct > self.exit_rank_pct:
                    sell_px = current_close * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    trade_logs.append({'trade_time': str(t), 'ts_code': code, 'action': 'RANK_EXIT_OUT_20%', 'price': sell_px, 'shares': shares, 'pnl': pnl})
                    exited_codes.append(code)
                    continue

                # C. 最多持有 90分钟 (6 根 K线) 或 14:45 强制清仓 (零隔夜)
                if pos['bars_held'] >= 6 or time_str >= '14:45':
                    sell_px = current_close * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    action = 'FORCE_CLOSE_1445' if time_str >= '14:45' else 'TIME_EXIT_90M'
                    trade_logs.append({'trade_time': str(t), 'ts_code': code, 'action': action, 'price': sell_px, 'shares': shares, 'pnl': pnl})
                    exited_codes.append(code)

            for code in exited_codes:
                if code in positions:
                    del positions[code]

            # 3. 09:45 ~ 14:30 截面轮动开仓
            if '09:45' <= time_str <= '14:30':
                open_slots = self.top_n - len(positions)
                if open_slots > 0 and capital > 10000.0:
                    candidates = []
                    for code, row in t_dict.items():
                        rank_15m = row.get('rank_15m', 999)
                        ret_15m = row.get('ret_15m', 0.0)
                        
                        # 暴涨过滤：单根 15m 涨幅 > 3.5% 禁止追高
                        if rank_15m <= self.top_n and ret_15m <= self.max_surge_cap and code not in positions:
                            candidates.append((code, row['close'], rank_15m))

                    candidates.sort(key=lambda x: x[2])

                    for code, close_px, r_rank in candidates:
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
                            trade_logs.append({'trade_time': str(t), 'ts_code': code, 'action': 'BUY', 'price': buy_px, 'shares': shares, 'pnl': 0.0})

        return pd.DataFrame(equity_curve), pd.DataFrame(trade_logs)
