# -*- coding: utf-8 -*-

"""
融合可转债量化策略：传统双低安全选底座 (Top 25) + LightGBM 机器学习二次置信度扫描与 ATR 风控
Hybrid Convertible Bond Strategy: Classic Double-Low Base Pool + LightGBM ML Enhancement
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

    return {
        'total_sells': total_sells,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_win': max_win,
        'max_loss': max_loss
    }

def main():
    logging.info("=== 开始运行 融合策略 (传统双低底座 + LightGBM 机器学习二次增强) ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2024-01-01", end_date="2026-07-25", max_bonds=250)
    
    # 1. 传统多因子选池 (筛选价格、溢价率、强赎及规模，生成双低 Top 25 安全候选池)
    logging.info("计算传统双低多因子与风控过滤...")
    df_trad = CBTraditionalFactorEngine.compute_traditional_factors(df_panel)
    df_pool = CBTraditionalFactorEngine.select_double_low_pool(df_trad, pool_size=25)
    
    # 2. 特征工程与 LightGBM 预测
    logging.info("在双低安全底座上构建微观体征与 LightGBM 预测特征...")
    df_ml_factors = CBMLFactorEngine.compute_ml_features(df_pool)
    
    train_mask = (df_ml_factors['trade_time'] >= '2024-01-01') & (df_ml_factors['trade_time'] < '2025-01-01')
    test_mask = (df_ml_factors['trade_time'] >= '2025-01-01')
    
    df_train = df_ml_factors[train_mask]
    df_test = df_ml_factors[test_mask]
    
    # 3. 仅在双低 Top 25 安全底座内训练并筛选 LightGBM 高置信度 (P >= 0.52)
    model_engine = CBMLModelEngine(n_estimators=150, learning_rate=0.03, max_depth=5, min_confidence=0.52)
    model_engine.train(df_train)
    
    # 限制模型仅在双低候选池内打分
    df_test_pool_only = df_test[df_test['in_double_low_pool'] == True].copy()
    df_predicted = model_engine.predict_ranks(df_test_pool_only)
    
    # 4. 融合策略回测引擎 (配合 1.2 * ATR 动态止损与 30 分钟冷却)
    strategy_engine = CBMLStrategyEngine(
        initial_capital=1000000.0,
        top_n=6,                 # 持仓 6 只
        exit_rank_k=12,          # 掉出前 12 名平仓
        atr_multiplier=1.2,      # 1.2 * ATR 动态止损
        max_hold_bars=18,        # 90 分钟时间平仓
        min_amp_threshold=0.006, # 振幅 >= 0.6% 覆盖成本
        cooldown_bars=6,         # 30 分钟买入冷却
        single_slippage=0.001,   # 0.1% 滑点
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
    print("  融合策略 (传统双低底座 + LightGBM ML 二次增强) 绩效分析")
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
    print(f"盈亏比 (Profit/Loss Ratio):      {trade_stats.get('profit_loss_ratio', 0.0):.2f}")
    print(f"平均单笔盈利:                   {trade_stats.get('avg_win', 0.0):,.2f} 元")
    print(f"平均单笔亏损:                   {trade_stats.get('avg_loss', 0.0):,.2f} 元")
    print(f"同期中证转债指数涨跌幅:           -10.38%")
    print(f"超越大盘超额收益 (Alpha):        {tot_ret*100 - (-10.38):+.2f}%")
    print("="*65 + "\n")
    
    if not df_equity.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(df_equity['trade_time']), df_equity['nav'], label='Hybrid Strategy NAV (Double-Low Base + LightGBM ML)', color='#9467bd', linewidth=2.0)
        plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
        plt.title("A-Share CB Hybrid Strategy (Double-Low Pool + LightGBM Enhancement)", fontsize=14)
        plt.xlabel("Trade Time", fontsize=12)
        plt.ylabel("Net Asset Value (RMB)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.savefig("cb_hybrid_strategy_equity.png", dpi=300)

if __name__ == "__main__":
    main()
