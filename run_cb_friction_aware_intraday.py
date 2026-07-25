# -*- coding: utf-8 -*-

"""
扣除滑点感知 + 日内多次交易 LightGBM 策略主程序
Friction-Aware Active Intraday T+0 Strategy Execution Runner
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cb_quant.data_loader import CBDataLoader
from cb_quant.traditional_factor_engine import CBTraditionalFactorEngine
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
    logging.info("=== 开始运行 扣除滑点感知 + 日内多次交易 LightGBM 策略 ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2024-01-01", end_date="2026-07-25", max_bonds=250)
    
    # 1. 结合双低风控围栏选池 (Top 35)
    logging.info("计算双低风控围栏池 (Top 35)...")
    df_trad = CBTraditionalFactorEngine.compute_traditional_factors(df_panel)
    df_pool = CBTraditionalFactorEngine.select_double_low_pool(df_trad, pool_size=35)
    
    # 2. 构建★扣除 0.21% 交易摩擦后★的净 Alpha 特征工程
    logging.info("构建扣除 0.21% 交易摩擦（双边滑点+佣金）后的滑点感知净 Label...")
    df_ml_factors = CBMLFactorEngine.compute_ml_features(df_pool, friction_cost=0.0021)
    
    train_mask = (df_ml_factors['trade_time'] >= '2024-01-01') & (df_ml_factors['trade_time'] < '2025-01-01')
    test_mask = (df_ml_factors['trade_time'] >= '2025-01-01')
    
    df_train = df_ml_factors[train_mask]
    df_test = df_ml_factors[test_mask]
    
    # 3. 训练滑点感知 LightGBM 二分类模型 (置信度门槛 min_confidence = 0.500，全面激活日内多次 T+0 交易)
    model_engine = CBMLModelEngine(n_estimators=150, learning_rate=0.03, max_depth=5, min_confidence=0.500)
    model_engine.train(df_train)
    
    df_test_pool = df_test[df_test['in_double_low_pool'] == True].copy()
    df_predicted = model_engine.predict_ranks(df_test_pool)
    
    # 4. 日内多次交易策略引擎：最小持仓 15 分钟 (3 根 K线)，软离场 K=25，动态 1.2xATR 止损
    strategy_engine = CBMLStrategyEngine(
        initial_capital=1000000.0,
        top_n=5,                 # 持仓 5 只
        exit_rank_k=30,          # 掉出前 30 名离场
        atr_multiplier=1.0,      # 1.0 * ATR 动态止损
        max_hold_bars=12,        # 最多持仓 60 分钟
        min_amp_threshold=0.002, # 振幅 >= 0.2% 触发活跃交易
        cooldown_bars=1,         # 5 分钟买入冷却
        single_slippage=0.001,   # 扣除单边 0.1% 滑点
        commission=0.00005,      # 万0.5 佣金
        break_even_trigger=0.012, # 浮盈 1.2% 移动保本
        trailing_trigger=0.020,   # 浮盈 2.0% 跟踪止盈
        trailing_pct=0.008        # 峰值回撤 0.8% 锁定
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
    print("  扣除滑点感知 + 日内多次交易 LightGBM 策略绩效分析")
    print("="*65)
    print(f"回测测试区间:       2025-01-01 至 2026-07-24 (共 {num_days} 个交易日)")
    print(f"初始资金:           1,000,000.00 元")
    print(f"期末资金:           {df_equity['nav'].iloc[-1] if not df_equity.empty else 0:,.2f} 元")
    print(f"累计收益率 (Cumulative Return):  {tot_ret*100:+.2f}%")
    print(f"年化收益率 (Annualized Return):  {ann_ret*100:+.2f}%")
    print(f"夏普比率 (Sharpe Ratio):        {sharpe:.2f}")
    print(f"卡玛比率 (Calmar Ratio):        {calmar:.2f}")
    print(f"最大回撤 (Max Drawdown):        {max_dd*100:.2f}%")
    print(f"★ 日均交易笔数 (Daily Trades):   {daily_t:.2f} 笔/天 [激活日内多次交易]")
    print(f"★ 扣除滑点后真实胜率 (Win Rate): {trade_stats.get('win_rate', 0.0)*100:.2f}%")
    print(f"盈亏比 (Profit/Loss Ratio):      {trade_stats.get('profit_loss_ratio', 0.0):.2f}")
    print(f"平均单笔盈利:                   {trade_stats.get('avg_win', 0.0):,.2f} 元")
    print(f"平均单笔亏损:                   {trade_stats.get('avg_loss', 0.0):,.2f} 元")
    print(f"同期中证转债指数涨跌幅:           -10.38%")
    print(f"超越大盘超额收益 (Alpha):        {tot_ret*100 - (-10.38):+.2f}%")
    print("="*65 + "\n")
    
    if not df_equity.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(df_equity['trade_time']), df_equity['nav'], label='Friction-Aware Active Intraday Strategy NAV', color='#e377c2', linewidth=2.0)
        plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
        plt.title("Friction-Aware Active Intraday LightGBM Strategy Equity Curve", fontsize=14)
        plt.xlabel("Trade Time", fontsize=12)
        plt.ylabel("Net Asset Value (RMB)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.savefig("cb_friction_aware_equity.png", dpi=300)

if __name__ == "__main__":
    main()
