# -*- coding: utf-8 -*-

"""
全量可转债 5 分钟线 LightGBM 正收益策略 (含市场择时风控与绝对超额收益)
A-Share CB LightGBM Strategy for Positive Absolute Return (with Market Trend Gate)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cb_quant.data_loader import CBDataLoader
from cb_quant.ml_factor_engine import CBMLFactorEngine
from cb_quant.ml_model import CBMLModelEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class PositiveCBStrategyEngine:
    def __init__(self, initial_capital=1000000.0, top_n=5, exit_rank_k=10,
                 atr_multiplier=1.2, max_hold_bars=36, min_amp_threshold=0.008,
                 cooldown_bars=6, single_slippage=0.001, commission=0.00005):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.exit_rank_k = exit_rank_k
        self.atr_multiplier = atr_multiplier
        self.max_hold_bars = max_hold_bars
        self.min_amp_threshold = min_amp_threshold
        self.cooldown_bars = cooldown_bars
        self.single_slippage = single_slippage
        self.commission = commission

    def run_backtest(self, df_predicted):
        df = df_predicted.copy()
        df['trade_time'] = df['trade_time'].astype(str)
        
        time_groups = dict(tuple(df.groupby('trade_time')))
        timestamps = sorted(time_groups.keys())
        
        capital = self.initial_capital
        positions = {}
        cooldown_dict = {}
        
        trade_logs = []
        equity_curve = []
        slot_target_value = capital / self.top_n

        for b_idx, t in enumerate(timestamps):
            df_t = time_groups[t]
            t_dict = df_t.set_index('ts_code').to_dict('index')
            
            # 计算全市场 15 分钟平均收益率作为大盘强弱择时 Gate
            market_15m_ret = df_t['market_ret_15m'].iloc[0] if 'market_ret_15m' in df_t.columns else 0.0
            
            # 1. 评估 NAV
            total_pos_val = 0.0
            for code, pos in positions.items():
                pos['bars_held'] += 1
                if code in t_dict:
                    px = t_dict[code]['close']
                    total_pos_val += pos['shares'] * px
                else:
                    total_pos_val += pos['shares'] * pos['entry_price']

            nav = capital + total_pos_val
            equity_curve.append({'trade_time': t, 'nav': nav, 'cash': capital, 'position_val': total_pos_val, 'num_positions': len(positions)})
            slot_target_value = nav / self.top_n

            # 2. 动态 ATR 止损检查
            atr_stopped = []
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                entry_px = pos['entry_price']
                current_low = row['low']
                stop_price = entry_px - pos['stop_dist']

                if current_low <= stop_price:
                    sell_px = current_low * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    pnl_pct = (sell_px / entry_px) - 1.0
                    trade_logs.append({
                        'trade_time': t, 'ts_code': code, 'action': 'ATR_STOP_LOSS',
                        'price': sell_px, 'shares': shares, 'pnl': pnl, 'pnl_pct': pnl_pct
                    })
                    atr_stopped.append(code)
                    cooldown_dict[code] = b_idx + self.cooldown_bars

            for code in atr_stopped:
                del positions[code]

            # 3. 软止损/排名退出与大盘恶化空仓防护
            rank_exited = []
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                rank = row.get('rank', 9999)
                
                # 若大盘突发急跌 (15m < -0.3%) 或 排名掉出前 K 名 -> 触发出局
                if (market_15m_ret < -0.003) or (pos['bars_held'] >= 6 and rank > self.exit_rank_k):
                    sell_px = row['close'] * (1.0 - self.single_slippage)
                    shares = pos['shares']
                    entry_px = pos['entry_price']
                    revenue = shares * sell_px * (1.0 - self.commission)
                    capital += revenue
                    
                    pnl = revenue - (shares * entry_px)
                    pnl_pct = (sell_px / entry_px) - 1.0
                    trade_logs.append({
                        'trade_time': t, 'ts_code': code, 'action': 'RANK_EXIT_SELL',
                        'price': sell_px, 'shares': shares, 'pnl': pnl, 'pnl_pct': pnl_pct
                    })
                    rank_exited.append(code)
                    cooldown_dict[code] = b_idx + self.cooldown_bars

            for code in rank_exited:
                del positions[code]

            # 4. 择时与置信度开仓 (只在大盘环境正常且有极高置信度标的时买入)
            if market_15m_ret >= -0.001:
                open_slots = self.top_n - len(positions)
                if open_slots > 0 and capital > 10000.0:
                    candidates = []
                    for code, row in t_dict.items():
                        rank = row.get('rank', 9999)
                        amp = row.get('amp_15m', 0.0)
                        
                        if rank <= self.top_n and amp >= self.min_amp_threshold and code not in positions and cooldown_dict.get(code, 0) <= b_idx:
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
                            stop_dist = max(buy_px * 0.012, self.atr_multiplier * atr_val)
                            
                            positions[code] = {
                                'shares': shares,
                                'entry_price': buy_px,
                                'entry_time': t,
                                'bars_held': 0,
                                'stop_dist': stop_dist
                            }
                            trade_logs.append({
                                'trade_time': t, 'ts_code': code, 'action': 'BUY',
                                'price': buy_px, 'shares': shares, 'pnl': 0.0, 'pnl_pct': 0.0
                            })

        return pd.DataFrame(equity_curve), pd.DataFrame(trade_logs)

def main():
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2024-01-01", end_date="2026-07-25", max_bonds=250)
    df_ml_factors = CBMLFactorEngine.compute_ml_features(df_panel)
    
    train_mask = (df_ml_factors['trade_time'] >= '2024-01-01') & (df_ml_factors['trade_time'] < '2025-01-01')
    test_mask = (df_ml_factors['trade_time'] >= '2025-01-01')
    
    df_train = df_ml_factors[train_mask]
    df_test = df_ml_factors[test_mask]
    
    model_engine = CBMLModelEngine(n_estimators=150, learning_rate=0.03, max_depth=5, min_confidence=0.52)
    model_engine.train(df_train)
    df_predicted = model_engine.predict_ranks(df_test)
    
    strategy_engine = PositiveCBStrategyEngine(
        initial_capital=1000000.0,
        top_n=5,
        exit_rank_k=10,
        atr_multiplier=1.2,
        max_hold_bars=36,
        min_amp_threshold=0.008,
        cooldown_bars=6,
        single_slippage=0.001,
        commission=0.00005
    )
    df_equity, df_trades = strategy_engine.run_backtest(df_predicted)
    
    if not df_equity.empty:
        nav_s = df_equity['nav']
        tot_ret = (nav_s.iloc[-1] - 1000000.0) / 1000000.0
        cummax = nav_s.cummax()
        mdd = ((nav_s - cummax) / cummax).min()
        sells = df_trades[df_trades['action'] != 'BUY']
        trades = len(sells)
        win = len(sells[sells['pnl'] > 0]) / max(trades, 1)
        
        print("\n" + "="*60)
        print("   可转债 LightGBM 正绝对收益策略 (含大盘择时) 绩效")
        print("="*60)
        print(f"初始资金:           1,000,000.00 元")
        print(f"期末资金:           {nav_s.iloc[-1]:,.2f} 元")
        print(f"绝对累计收益率:     {tot_ret*100:+.2f}%")
        print(f"最大回撤 (MaxDD):   {mdd*100:.2f}%")
        print(f"平仓交易胜率:       {win*100:.2f}%")
        print(f"平仓交易次数:       {trades} 次 (日均仅 {trades/377.0:.2f} 次)")
        print("="*60 + "\n")
        
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(df_equity['trade_time']), df_equity['nav'], label='Positive Absolute Return Strategy NAV', color='#2ca02c', linewidth=2.0)
        plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
        plt.title("A-Share CB LightGBM Positive Absolute Return Strategy Equity Curve", fontsize=14)
        plt.xlabel("Trade Time", fontsize=12)
        plt.ylabel("Net Asset Value (RMB)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.savefig("cb_positive_strategy_equity.png", dpi=300)

if __name__ == "__main__":
    main()
