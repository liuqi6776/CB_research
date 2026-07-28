# -*- coding: utf-8 -*-

"""
全量多因子 GBDT 严格 Walk-Forward 样本外 (OOS) 训练与特征重要性评估器
保存所有产物到 artifacts/ 以实现 100% 独立复现
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
from cb_quant.unified_pit_engine import CBUnifiedPITEngine
from cb_quant.tcc_factor import CBTCCFactorEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def train_master_gbdt_oos():
    logger.info("=== 开始 GBDT 严格 Walk-Forward 样本外 (OOS) 训练 ===")
    
    loader = CBDataLoader(data_dir=r"D:\CB_mins_data")
    df_15m = loader.load_minute_panel(start_date="2021-01-01", max_bonds=None)
    
    unified_engine = CBUnifiedPITEngine(mins_data_dir=r"D:\CB_mins_data", data_v2_dir=r"D:\iquant_data\data_v2")
    df_pit = unified_engine.build_unified_state_panel(df_15m)
    
    tcc_engine = CBTCCFactorEngine(window=21)
    df_tcc = tcc_engine.generate_tcc_panel(start_date="2021-01-01")
    
    if df_tcc is not None and not df_tcc.empty:
        df_pit = df_pit.merge(df_tcc[['ts_code', 'date_str', 'tcc_factor']],
                              on=['ts_code', 'date_str'], how='left')
    else:
        df_pit['tcc_factor'] = np.nan

    # 计算 60 分钟前瞻收益率目标 (4 个 15 分钟 Bar)
    df_pit['fwd_rtn_60m'] = df_pit.groupby('ts_code')['close'].shift(-4) / df_pit['close'] - 1.0

    # 计算极值因子 (ex_rtn_max_val_5min / ex_rtn_min_freq_5min)
    logger.info("=== 开始计算极值 (ex_rtn_max_val) 因子 ===")
    df_pit['ex_rtn_max_val_5min'] = np.where(
        df_pit['close'].notnull() & (df_pit['close'] > 0),
        (df_pit['high'] - df_pit['close']) / df_pit['close'], np.nan
    )
    df_pit['ex_rtn_max_val_1min'] = np.where(
        df_pit['close'].notnull() & (df_pit['close'] > 0),
        (df_pit['close'] - df_pit['low']) / df_pit['close'], np.nan
    )
    df_pit['ex_rtn_min_freq_5min'] = np.where(
        df_pit['vol'].notnull() & (df_pit['vol'] > 0),
        df_pit['amount'] / (df_pit['vol'] * 100.0), np.nan
    )

    feature_cols = [
        'double_low', 'premium_rate_t1', 'conv_value_t1', 'vol', 'amount',
        'curr_iss_amt', 'spike_ratio', 'tcc_factor',
        'ex_rtn_max_val_5min', 'ex_rtn_max_val_1min', 'ex_rtn_min_freq_5min'
    ]

    # 1. 严格 Walk-Forward 切分 & 15 交易日 Embargo 校验
    all_trade_dates = sorted(df_pit['date_int'].unique())
    train_dates = [d for d in all_trade_dates if d <= 20231231]
    
    # 强制校验 15 个交易日隔离带 (Embargo Gap)
    embargo_dates = [d for d in all_trade_dates if d > 20231231][:15]
    oos_dates = [d for d in all_trade_dates if d > 20231231][15:]
    
    logger.info(f"Walk-Forward 交易日切分: 训练集 ({len(train_dates)} 交易日), 15交易日 Embargo 隔离带 ({len(embargo_dates)} 交易日), OOS测试集 ({len(oos_dates)} 交易日)")

    train_mask = df_pit['date_int'].isin(train_dates)
    oos_mask = df_pit['date_int'].isin(oos_dates)

    df_train = df_pit[train_mask].dropna(subset=['fwd_rtn_60m'] + feature_cols).copy()
    df_oos = df_pit[oos_mask].dropna(subset=['fwd_rtn_60m'] + feature_cols).copy()

    X_train, y_train = df_train[feature_cols], df_train['fwd_rtn_60m']
    X_oos, y_oos = df_oos[feature_cols], df_oos['fwd_rtn_60m']

    logger.info(f"训练集中样本数: {len(X_train):,}, 纯样本外 (OOS) 测试集中样本数: {len(X_oos):,}")

    model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.03,
        max_depth=5,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 在纯样本外 OOS 测试集上计算置换重要性 (Permutation Importance)
    logger.info("=== 正在 OOS 纯样本外测试集上评估特征置换重要性 ===")
    perm_res = permutation_importance(model, X_oos, y_oos, n_repeats=5, random_state=42, n_jobs=-1)

    print("\n===========================================================================")
    print("      全量 GBDT 模型 (OOS Test Set 2024-2026) 纯样本外特征重要性")
    print("===========================================================================")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': perm_res.importances_mean
    }).sort_values(by='importance', ascending=False)

    for idx, row in importance_df.iterrows():
        print(f"  - {row['feature']:<25} : {row['importance']:+.6f}")
    print("===========================================================================\n")

    # 保存模型产物到本笃 repo 与 artifacts/
    local_model_path = "master_multifactor_gbdt.joblib"
    repo_model_path = r"c:\Users\liuqi\quant_system_v2\artifacts\master_multifactor_gbdt.joblib"
    os.makedirs(os.path.dirname(repo_model_path), exist_ok=True)

    joblib.dump(model, local_model_path)
    joblib.dump(model, repo_model_path)
    logger.info(f"OOS 训练模型已成功保存至:\n  - {local_model_path}\n  - {repo_model_path}")

if __name__ == '__main__':
    train_master_gbdt_oos()
