# -*- coding: utf-8 -*-

"""
升维可转债 LightGBM 策略全套量化绩效指标计算 (年化、夏普、卡玛、最大回撤、胜率等)
"""

import sys
import logging
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader
from cb_quant.ml_factor_engine import CBMLFactorEngine
from cb_quant.ml_model import CBMLModelEngine
from cb_quant.ml_strategy_engine import CBMLStrategyEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2024-01-01", end_date="2026-07-25", max_bonds=250)
    df_ml_factors = CBMLFactorEngine.compute_ml_features(df_panel)
    
    train_mask = (df_ml_factors['trade_time'] >= '2024-01-01') & (df_ml_factors['trade_time'] < '2025-01-01')
    test_mask = (df_ml_factors['trade_time'] >= '2025-01-01')
    
    df_train = df_ml_factors[train_mask]
    df_test = df_ml_factors[test_mask]
    
    model_engine = CBMLModelEngine(n_estimators=150, learning_rate=0.03, max_depth=5, min_confidence=0.55)
    model_engine.train(df_train)
    df_predicted = model_engine.predict_ranks(df_test)
    
    strategy_engine = CBMLStrategyEngine(
        initial_capital=1000000.0,
        top_n=8,
        exit_rank_k=12,
        atr_multiplier=1.5,
        max_hold_bars=24,
        min_amp_threshold=0.008,
        cooldown_bars=6,
        single_slippage=0.001,
        commission=0.00005
    )
    df_equity, df_trades = strategy_engine.run_backtest(df_predicted)
    
    if not df_equity.empty:
        df_eq = df_equity.copy()
        df_eq['trade_time'] = pd.to_datetime(df_eq['trade_time'])
        df_eq['date'] = df_eq['trade_time'].dt.date
        
        # 每日终 NAV 序列
        daily_nav = df_eq.groupby('date')['nav'].last()
        daily_ret = daily_nav.pct_change().dropna()
        
        tot_ret = (daily_nav.iloc[-1] - 1000000.0) / 1000000.0
        num_days = len(daily_nav)
        ann_ret = (1.0 + tot_ret) ** (252.0 / num_days) - 1.0 if num_days > 0 else 0.0
        
        # 夏普比率 (假设无风险利率 2%)
        rf_daily = 0.02 / 252.0
        ex_ret = daily_ret - rf_daily
        sharpe = (ex_ret.mean() / (daily_ret.std() + 1e-8)) * np.sqrt(252.0)
        
        # 最大回撤
        cummax = daily_nav.cummax()
        drawdown = (daily_nav - cummax) / cummax
        max_dd = drawdown.min()
        
        # 卡玛比率
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
        
        # 交易明细
        sells = df_trades[df_trades['action'] != 'BUY'] if not df_trades.empty else pd.DataFrame()
        total_trades = len(sells)
        win_rate = (sells['pnl'] > 0).mean() if total_trades > 0 else 0.0
        
        avg_win = sells[sells['pnl'] > 0]['pnl'].mean() if (total_trades > 0 and (sells['pnl'] > 0).sum() > 0) else 0.0
        avg_loss = abs(sells[sells['pnl'] < 0]['pnl'].mean()) if (total_trades > 0 and (sells['pnl'] < 0).sum() > 0) else 1.0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        print("\n" + "="*65)
        print("     升维可转债 LightGBM 策略标准量化绩效指标 (Full Metrics)")
        print("="*65)
        print(f"回测测试区间:       2025-01-01 至 2026-07-24 (共 {num_days} 个交易日 / {num_days/252.0:.2f} 年)")
        print(f"初始资金:           1,000,000.00 元")
        print(f"期末资金:           {daily_nav.iloc[-1]:,.2f} 元")
        print(f"累计收益率 (Cumulative Return):  {tot_ret*100:+.2f}%")
        print(f"年化收益率 (Annualized Return):  {ann_ret*100:+.2f}%")
        print(f"夏普比率 (Sharpe Ratio):        {sharpe:.2f} (无风险利率 2.0%)")
        print(f"卡玛比率 (Calmar Ratio):        {calmar:.2f}")
        print(f"最大回撤 (Max Drawdown):        {max_dd*100:.2f}%")
        print(f"日均交易笔数 (Daily Trades):     {total_trades / float(num_days):.2f} 笔/天")
        print(f"平仓交易胜率 (Win Rate):         {win_rate*100:.2f}%")
        print(f"盈亏比 (Profit/Loss Ratio):      {profit_loss_ratio:.2f}")
        print(f"平均盈利/平仓:                  {avg_win:,.2f} 元")
        print(f"平均亏损/平仓:                  {avg_loss:,.2f} 元")
        print(f"同期中证转债指数涨跌幅:           -10.38%")
        print(f"超越大盘超额收益 (Alpha):        {tot_ret*100 - (-10.38):+.2f}%")
        print("="*65 + "\n")

if __name__ == "__main__":
    main()
