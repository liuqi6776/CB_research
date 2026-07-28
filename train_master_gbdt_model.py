# -*- coding: utf-8 -*-

"""
全量多因子 GBDT 严格样本外 (Walk-Forward OOS) 训练器
Walk-Forward Out-of-Sample GBDT Model Trainer with Temporal Embargo Gap

规则防范:
1. 严禁全样本 fit! 训练集限定于 2020-01-01 ~ 2023-12-31;
2. 插入 15 个交易日 Embargo 隔离带，防范重叠收益率（fut_ret_60m_close）信息渗漏;
3. 样本外 (OOS Test Set: 2024-01-01 ~ 2026-07-25) 仅用于推理与预测;
4. 特征置换重要性 (Permutation Importance) 严格在 OOS 测试集上计算。
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

def train_master_gbdt_oos():
    logging.info("=== 启动【全量多因子 GBDT 严格样本外 Walk-Forward】训练流程 ===")
    
    loader = CBDataLoader()
    # 覆盖 2021 ~ 2026 全长数据，使用确定性排序宇宙
    df_panel = loader.load_minute_panel(start_date="2021-01-01", end_date="2026-07-25", max_bonds=None)
    
    clean_engine = CBStrict15mCleanEngine()
    df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
    
    pit_adapter = CBAsOfPITAdapter()
    df_15m = pit_adapter.attach_asof_pit_metadata(df_15m)

    unified_engine = CBUnifiedPITEngine()
    df_pit = unified_engine.build_unified_state_panel(df_15m)
    
    # 接入 TCC 因子
    tcc_engine = CBTCCFactorEngine(window=21)
    tcc_long = tcc_engine.generate_tcc_panel(start_date="2020-01-01", end_date="2026-07-25")
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

    # Walk-Forward 切分: 训练集 <= 2023-12-31, OOS 测试集 >= 2024-01-20 (中间留出 15 天 Embargo)
    train_mask = (df_clean['date_str'] <= '20231231')
    oos_mask = (df_clean['date_str'] >= '20240120')

    X_train = df_clean.loc[train_mask, feature_cols].fillna(0.0)
    y_train = df_clean.loc[train_mask, 'fut_ret_60m_close']

    X_oos = df_clean.loc[oos_mask, feature_cols].fillna(0.0)
    y_oos = df_clean.loc[oos_mask, 'fut_ret_60m_close']

    logging.info(f"Walk-Forward 切分完成: 训练集 ({X_train.shape[0]:,} 行), Embargo (15天隔离), OOS测试集 ({X_oos.shape[0]:,} 行)。")
    
    # 严格仅在训练集 fit 模型
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.03,
        max_depth=5,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 样本外 (OOS) 置换重要性评估
    sample_size = min(20000, len(X_oos))
    perm_imp = permutation_importance(model, X_oos.iloc[:sample_size], y_oos.iloc[:sample_size], n_repeats=5, random_state=42)
    imp_series = pd.Series(perm_imp.importances_mean, index=feature_cols).sort_values(ascending=False)
    
    print("\n" + "="*75)
    print("      【全量多因子 GBDT 样本外 (OOS Test Set 2024-2026) 置换重要性】")
    print("="*75)
    for feat, imp in imp_series.items():
        print(f"  - {feat:25s}: {imp:+.6f}")
    print("="*75 + "\n")

    # 保存 OOS 专用模型文件
    model_file = "master_multifactor_gbdt.joblib"
    joblib.dump(model, model_file)
    logging.info(f"OOS 训练模型成功，已保存至: {model_file}")

if __name__ == '__main__':
    train_master_gbdt_oos()
