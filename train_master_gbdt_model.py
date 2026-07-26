# -*- coding: utf-8 -*-

"""
全量多因子 GBDT 模型训练器 (Master Multi-Factor GBDT Model Trainer)
融合全部已验证因子线:
1. double_low: 双低得分
2. conv_value_t1: T-1 转股价值
3. premium_rate_t1: T-1 转股溢价率
4. curr_iss_amt: 债券剩余规模
5. tcc_factor: TCC 时间网络相对中心度
6. ex_rtn_max_val_5min: 5分钟收益率极大值幅度
7. ex_rtn_max_val_1min: 1分钟收益率极大值幅度
8. ex_rtn_min_freq_5min: 5分钟极小值频率
9. vol / amount / spike_ratio: 15m 盘中量价动量
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from cb_quant.data_loader import CBDataLoader
from cb_quant.strict_15m_clean_engine import CBStrict15mCleanEngine
from cb_quant.unified_pit_engine import CBUnifiedPITEngine
from cb_quant.asof_pit_adapter import CBAsOfPITAdapter
from cb_quant.tcc_factor import CBTCCFactorEngine
from cb_quant.extreme_return_factor import CBExtremeReturnFactorEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_master_gbdt():
    logging.info("=== 启动【全量多因子 GBDT 机器学习模型】训练流程 ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    clean_engine = CBStrict15mCleanEngine()
    df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
    
    pit_adapter = CBAsOfPITAdapter()
    df_15m = pit_adapter.attach_asof_pit_metadata(df_15m)

    unified_engine = CBUnifiedPITEngine()
    df_pit = unified_engine.build_unified_state_panel(df_15m)
    
    # 接入 TCC 因子
    tcc_engine = CBTCCFactorEngine(window=21)
    tcc_long = tcc_engine.generate_tcc_panel(start_date="2024-12-01", end_date="2026-07-25")
    u_tcc_dates = sorted(tcc_long['date_str'].unique())
    map_tcc = {u_tcc_dates[i]: u_tcc_dates[i+1] for i in range(len(u_tcc_dates)-1)}
    tcc_long['t1_trade_date'] = tcc_long['date_str'].map(map_tcc)
    
    df_pit = df_pit.merge(tcc_long[['ts_code', 't1_trade_date', 'tcc_factor']],
                          left_on=['ts_code', 'date_str'], right_on=['ts_code', 't1_trade_date'], how='left')
                          
    # 接入收益率极大值幅度因子
    erm_engine = CBExtremeReturnFactorEngine()
    df_erm = erm_engine.generate_extreme_return_panel(df_panel)
    u_erm_dates = sorted(df_erm['date_str'].unique())
    map_erm = {u_erm_dates[i]: u_erm_dates[i+1] for i in range(len(u_erm_dates)-1)}
    df_erm['t1_trade_date'] = df_erm['date_str'].map(map_erm)
    
    df_pit = df_pit.merge(df_erm[['ts_code', 't1_trade_date', 'ex_rtn_max_val_5min', 'ex_rtn_max_val_1min', 'ex_rtn_min_freq_5min']],
                          left_on=['ts_code', 'date_str'], right_on=['ts_code', 't1_trade_date'], how='left')

    feature_cols = [
        'double_low', 'conv_value_t1', 'premium_rate_t1', 'curr_iss_amt',
        'tcc_factor', 'ex_rtn_max_val_5min', 'ex_rtn_max_val_1min', 'ex_rtn_min_freq_5min',
        'spike_ratio', 'vol', 'amount'
    ]
    
    df_clean = df_pit[
        (df_pit['is_eligible_at_selection'] == True) &
        (df_pit['fut_ret_60m_close'].notnull())
    ].copy()

    X = df_clean[feature_cols].fillna(0.0)
    y = df_clean['fut_ret_60m_close']

    logging.info(f"清洗后的训练样本集维度: {X.shape[0]:,} 行, {X.shape[1]} 个全量因子特征。")
    
    # 训练 HistGradientBoostingRegressor (GBDT 树模型)
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.03,
        max_depth=5,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    )
    model.fit(X, y)
    
    # 置换重要性评估
    perm_imp = permutation_importance(model, X.iloc[:20000], y.iloc[:20000], n_repeats=5, random_state=42)
    imp_series = pd.Series(perm_imp.importances_mean, index=feature_cols).sort_values(ascending=False)
    
    print("\n" + "="*70)
    print("      【全量多因子 GBDT 机器学习特征置换重要性 (Permutation Importance)】")
    print("="*70)
    for feat, imp in imp_series.items():
        print(f"  - {feat:25s}: {imp:+.6f}")
    print("="*70 + "\n")

    # 保存模型
    model_file = "master_multifactor_gbdt.joblib"
    joblib.dump(model, model_file)
    logging.info(f"模型训练成功，已保存至: {model_file}")

if __name__ == '__main__':
    train_master_gbdt()
