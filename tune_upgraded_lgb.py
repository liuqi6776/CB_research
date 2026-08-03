# -*- coding: utf-8 -*-

"""
升维 LightGBM 策略微调与正收益探索
Upgraded LightGBM Strategy Fine-Tuning for Positive Expectation
"""

import sys
import logging
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
    
    print("=== 开始升维 LightGBM 参数微调 ===")
    
    for conf in [0.50, 0.52, 0.55]:
        model_engine = CBMLModelEngine(n_estimators=150, learning_rate=0.03, max_depth=5, min_confidence=conf)
        model_engine.train(df_train)
        df_predicted = model_engine.predict_ranks(df_test)
        
        for atr_mult in [1.0, 1.2, 1.5]:
            for top_n in [5, 8]:
                strategy_engine = CBMLStrategyEngine(
                    initial_capital=1000000.0,
                    top_n=top_n,
                    exit_rank_k=top_n + 4,
                    atr_multiplier=atr_mult,
                    max_hold_bars=24,
                    min_amp_threshold=0.008,
                    cooldown_bars=6,
                    single_slippage=0.001,
                    commission=0.00005
                )
                df_eq, df_tr = strategy_engine.run_backtest(df_predicted)
                
                if not df_eq.empty:
                    nav_s = df_eq['nav']
                    ret = (nav_s.iloc[-1] - 1000000.0) / 1000000.0
                    cummax = nav_s.cummax()
                    mdd = ((nav_s - cummax) / cummax).min()
                    sells = df_tr[df_tr['action'] != 'BUY']
                    trades = len(sells)
                    win = len(sells[sells['pnl'] > 0]) / max(trades, 1)
                    
                    print(f"Conf: {conf:.2f} | ATR: {atr_mult:.1f}x | N: {top_n} | 累计收益: {ret*100:6.2f}% | 最大回撤: {mdd*100:6.2f}% | 胜率: {win*100:5.1f}% | 平仓次数: {trades}")

if __name__ == "__main__":
    main()
