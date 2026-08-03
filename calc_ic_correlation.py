# -*- coding: utf-8 -*-

"""
计算 LightGBM 预测置信度与未来收益率的相关系数 (IC & Rank IC)
兼测试日内高频 T+0 交易 (10-20分钟短持仓) 的收益表现
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

def calculate_ic_metrics(df_predicted):
    """计算置信度得分 prob_upward 与实际未来 60分钟收益率 fut_ret_60m 的 IC / Rank IC"""
    df_clean = df_predicted.dropna(subset=['prob_upward', 'fut_ret_60m']).copy()
    
    # 逐 5 分钟时间节点计算横截面 IC
    def calc_cross_sectional_ic(g):
        if len(g) < 10 or g['prob_upward'].std() == 0:
            return pd.Series({'ic': np.nan, 'rank_ic': np.nan})
        ic = g['prob_upward'].corr(g['fut_ret_60m'], method='pearson')
        rank_ic = g['prob_upward'].corr(g['fut_ret_60m'], method='spearman')
        return pd.Series({'ic': ic, 'rank_ic': rank_ic})

    ic_df = df_clean.groupby('trade_time').apply(calc_cross_sectional_ic).dropna()
    
    mean_ic = ic_df['ic'].mean()
    mean_rank_ic = ic_df['rank_ic'].mean()
    ic_std = ic_df['ic'].std()
    icir = mean_ic / (ic_std + 1e-8) * np.sqrt(252.0 * 48) # 年化 ICIR
    
    return mean_ic, mean_rank_ic, icir

def main():
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2024-01-01", end_date="2026-07-25", max_bonds=250)
    df_ml_factors = CBMLFactorEngine.compute_ml_features(df_panel)
    
    train_mask = (df_ml_factors['trade_time'] >= '2024-01-01') & (df_ml_factors['trade_time'] < '2025-01-01')
    test_mask = (df_ml_factors['trade_time'] >= '2025-01-01')
    
    df_train = df_ml_factors[train_mask]
    df_test = df_ml_factors[test_mask]
    
    # 1. 训练 LightGBM 模型
    model_engine = CBMLModelEngine(n_estimators=150, learning_rate=0.03, max_depth=5, min_confidence=0.51)
    model_engine.train(df_train)
    df_predicted = model_engine.predict_ranks(df_test)
    
    # 2. 计算 IC 与 Rank IC
    mean_ic, mean_rank_ic, icir = calculate_ic_metrics(df_predicted)
    
    print("\n" + "="*65)
    print("  LightGBM 置信度得分与实际未来收益率相关系数分析 (IC / Rank IC)")
    print("="*65)
    print(f"横截面平均 IC (Pearson 线性相关性):    {mean_ic:+.4f}")
    print(f"横截面平均 Rank IC (Spearman 秩相关性): {mean_rank_ic:+.4f}")
    print(f"年化 ICIR (信息比率):                 {icir:+.2f}")
    print("="*65 + "\n")
    
    # 3. 日内多次交易 T+0 探索 (最小持仓 2 根 K 线 = 10分钟，高敏触发日内冲高)
    print("=== 开始测试日内多次交易 T+0 策略 (10~20分钟活跃轮换) ===")
    
    for conf in [0.51, 0.52]:
        for min_hold in [2, 3]: # 10分钟/15分钟短持仓
            strategy_engine = CBMLStrategyEngine(
                initial_capital=1000000.0,
                top_n=6,
                exit_rank_k=10,
                atr_multiplier=1.2,
                max_hold_bars=min_hold,
                min_amp_threshold=0.006, # 降门槛提频
                cooldown_bars=4,         # 20 分钟买入冷却
                single_slippage=0.0005,  # 贴近盘口五档 0.05% 滑点
                commission=0.00005
            )
            df_eq, df_tr = strategy_engine.run_backtest(df_predicted)
            
            if not df_eq.empty:
                nav_s = df_eq['nav']
                tot_ret = (nav_s.iloc[-1] - 1000000.0) / 1000000.0
                cummax = nav_s.cummax()
                mdd = ((nav_s - cummax) / cummax).min()
                sells = df_tr[df_tr['action'] != 'BUY']
                trades = len(sells)
                win = len(sells[sells['pnl'] > 0]) / max(trades, 1)
                daily_t = trades / 377.0
                
                print(f"Conf: {conf:.2f} | Hold: {min_hold*5}m | 累计收益: {tot_ret*100:6.2f}% | 最大回撤: {mdd*100:6.2f}% | 胜率: {win*100:5.1f}% | 日均交易: {daily_t:.1f}次/天")

if __name__ == "__main__":
    main()
