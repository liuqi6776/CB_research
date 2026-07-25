# -*- coding: utf-8 -*-

"""
三层主动止盈 + 大盘择时 升维可转债 LightGBM 策略主程序
Main Execution: Three-Layer Active Take-Profit & Market Timing Strategy
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
from cb_quant.ml_strategy_engine import CBMLStrategyEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def analyze_trades_detail(df_trades):
    if df_trades.empty or 'action' not in df_trades.columns:
        return {}
        
    sells = df_trades[df_trades['action'] != 'BUY'].copy()
    if sells.empty:
        return {}

    total_sells = len(sells)
    wins = sells[sells['pnl'] > 0]
    losses = sells[sells['pnl'] < 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_sells if total_sells > 0 else 0.0

    avg_win = wins['pnl'].mean() if win_count > 0 else 0.0
    avg_loss = abs(losses['pnl'].mean()) if loss_count > 0 else 1.0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    max_win = sells['pnl'].max()
    max_loss = sells['pnl'].min()

    # 离场类型分布统计
    action_counts = sells['action'].value_counts().to_dict()

    return {
        'total_sells': total_sells,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_win': max_win,
        'max_loss': max_loss,
        'action_counts': action_counts
    }

def main():
    logging.info("=== 开始运行 三层主动止盈 + 大盘择时 升维可转债策略 ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2024-01-01", end_date="2026-07-25", max_bonds=250)
    
    logging.info("构建特征库与三层止盈目标 Label...")
    df_ml_factors = CBMLFactorEngine.compute_ml_features(df_panel)
    
    train_mask = (df_ml_factors['trade_time'] >= '2024-01-01') & (df_ml_factors['trade_time'] < '2025-01-01')
    test_mask = (df_ml_factors['trade_time'] >= '2025-01-01')
    
    df_train = df_ml_factors[train_mask]
    df_test = df_ml_factors[test_mask]
    
    # 训练 LightGBM 二分类模型 (置信度门槛 min_confidence = 0.52)
    model_engine = CBMLModelEngine(n_estimators=150, learning_rate=0.03, max_depth=5, min_confidence=0.52)
    model_engine.train(df_train)
    
    df_predicted = model_engine.predict_ranks(df_test)
    
    # 策略引擎：包含移动保本 (+1.5%)、动态跟踪止盈 (+2.5% 触达后峰值回撤 1.2% 锁定)、目标时间止盈 (+1.0%)
    strategy_engine = CBMLStrategyEngine(
        initial_capital=1000000.0,
        top_n=6,                 # 持仓 6 只
        exit_rank_k=18,          # 掉出前 18 名才软平仓，给盈利奔跑留足空间
        atr_multiplier=1.2,      # 1.2 * ATR 动态止损
        max_hold_bars=24,        # 120 分钟时间平仓
        min_amp_threshold=0.008, # 振幅 >= 0.8%
        cooldown_bars=4,         # 20 分钟买入冷却
        single_slippage=0.001,   # 0.1% 滑点
        commission=0.00005,      # 万0.5 佣金
        break_even_trigger=0.010, # 浮盈 1.0% 触发移动保本
        trailing_trigger=0.015,   # 浮盈 1.5% 触发跟踪止盈
        trailing_pct=0.006        # 峰值回撤 0.6% 锁定
    )
    
    df_equity, df_trades = strategy_engine.run_backtest(df_predicted)
    
    trade_stats = analyze_trades_detail(df_trades)
    
    if not df_equity.empty:
        df_eq = df_equity.copy()
        df_eq['trade_time'] = pd.to_datetime(df_eq['trade_time'])
        df_eq['date'] = df_eq['trade_time'].dt.date
        
        daily_nav = df_eq.groupby('date')['nav'].last()
        daily_ret = daily_nav.pct_change().dropna()
        
        tot_ret = (daily_nav.iloc[-1] - 1000000.0) / 1000000.0
        num_days = len(daily_nav)
        ann_ret = (1.0 + tot_ret) ** (252.0 / num_days) - 1.0 if num_days > 0 else 0.0
        
        rf_daily = 0.02 / 252.0
        ex_ret = daily_ret - rf_daily
        sharpe = (ex_ret.mean() / (daily_ret.std() + 1e-8)) * np.sqrt(252.0)
        
        cummax = daily_nav.cummax()
        drawdown = (daily_nav - cummax) / cummax
        max_dd = drawdown.min()
        
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
        total_trades = trade_stats.get('total_sells', 0)
        daily_t = total_trades / float(num_days) if num_days > 0 else 0.0
    else:
        tot_ret, ann_ret, sharpe, max_dd, calmar, daily_t, num_days = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        
    print("\n" + "="*65)
    print("  三层主动止盈 + 大盘择时 升维可转债 LightGBM 策略标准量化绩效分析")
    print("="*65)
    print(f"回测测试区间:       2025-01-01 至 2026-07-24 (共 {num_days} 个交易日)")
    print(f"初始资金:           1,000,000.00 元")
    print(f"期末资金:           {df_equity['nav'].iloc[-1] if not df_equity.empty else 0:,.2f} 元")
    print(f"累计收益率 (Cumulative Return):  {tot_ret*100:+.2f}%")
    print(f"年化收益率 (Annualized Return):  {ann_ret*100:+.2f}%")
    print(f"夏普比率 (Sharpe Ratio):        {sharpe:.2f}")
    print(f"卡玛比率 (Calmar Ratio):        {calmar:.2f}")
    print(f"最大回撤 (Max Drawdown):        {max_dd*100:.2f}%")
    print(f"日均交易笔数 (Daily Trades):     {daily_t:.2f} 笔/天")
    print(f"平仓交易胜率 (Win Rate):         {trade_stats.get('win_rate', 0.0)*100:.2f}%")
    print(f"★ 盈亏比 (Profit/Loss Ratio):   {trade_stats.get('profit_loss_ratio', 0.0):.2f} (目标 > 1.0)")
    print(f"平均单笔盈利:                   {trade_stats.get('avg_win', 0.0):,.2f} 元")
    print(f"平均单笔亏损:                   {trade_stats.get('avg_loss', 0.0):,.2f} 元")
    print(f"离场类型分布:                   {trade_stats.get('action_counts', {})}")
    print("="*65 + "\n")
    
    if not df_equity.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(df_equity['trade_time']), df_equity['nav'], label='3-Layer Take-Profit Strategy NAV', color='#ff7f0e', linewidth=2.0)
        plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
        plt.title("A-Share CB 3-Layer Active Take-Profit Strategy Equity Curve", fontsize=14)
        plt.xlabel("Trade Time", fontsize=12)
        plt.ylabel("Net Asset Value (RMB)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.savefig("cb_threelayer_strategy_equity.png", dpi=300)

if __name__ == "__main__":
    main()
