# -*- coding: utf-8 -*-

"""
严格 PIT 管道下的 LightGBM 模型重训练脚本 (LightGBM Re-Training on Strict PIT Pipeline)
确保所有特征与目标均为 As-Of 对齐与延迟成交，彻底消除前视偏差。
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cb_quant.data_loader import CBDataLoader
from cb_quant.strict_15m_clean_engine import CBStrict15mCleanEngine
from cb_quant.unified_pit_engine import CBUnifiedPITEngine
from cb_quant.asof_pit_adapter import CBAsOfPITAdapter
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class StrictPITLGBTrainer:
    """
    严格 As-Of PIT GBDT/LightGBM 重训练器
    """
    def __init__(self):
        self.feature_cols = [
            'double_low', 'premium_rate_t1', 'conv_value_t1', 'curr_iss_amt',
            'chip_profit_ratio', 'chip_concentration_90', 'chip_position_20d',
            'spike_ratio', 'vol', 'amount'
        ]

    def build_feature_matrix(self):
        logging.info("=== 开始构建严格 As-Of PIT 无前视特征矩阵 ===")
        loader = CBDataLoader()
        df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
        
        clean_engine = CBStrict15mCleanEngine()
        df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
        
        pit_adapter = CBAsOfPITAdapter()
        df_15m = pit_adapter.attach_asof_pit_metadata(df_15m)

        unified_engine = CBUnifiedPITEngine()
        df_pit = unified_engine.build_unified_state_panel(df_15m)
        
        # 仅选择选债合规 (is_eligible_at_selection) 且处于有效信号窗口的样本
        df_valid = df_pit[
            (df_pit['is_eligible_at_selection'] == True) &
            (df_pit['fut_ret_60m_close'].notnull())
        ].copy()
        
        for c in self.feature_cols:
            if c not in df_valid.columns:
                df_valid[c] = np.nan
                
        df_valid.dropna(subset=['double_low', 'fut_ret_60m_close'], inplace=True)
        logging.info(f"特征矩阵构建完毕，有效训练/验证样本共 {len(df_valid):,} 行。")
        return df_valid

    def train_model(self, df_matrix, model_path="lgb_strict_pit_model.joblib"):
        logging.info("=== 训练 HistGradientBoosting GBDT 模型 ===")
        
        train_mask = df_matrix['date_str'] < '20250701'
        test_mask = df_matrix['date_str'] >= '20250701'

        X_train, y_train = df_matrix.loc[train_mask, self.feature_cols], df_matrix.loc[train_mask, 'fut_ret_60m_close']
        X_test, y_test = df_matrix.loc[test_mask, self.feature_cols], df_matrix.loc[test_mask, 'fut_ret_60m_close']

        logging.info(f"样本分布: 训练集 {len(X_train):,} | 测试集 {len(X_test):,}")

        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.03,
            max_leaf_nodes=15,
            max_depth=4,
            random_state=42
        )
        
        if not X_train.empty:
            model.fit(X_train, y_train)
        else:
            model.fit(df_matrix[self.feature_cols], df_matrix['fut_ret_60m_close'])

        joblib.dump(model, model_path)
        logging.info(f"严格 As-Of PIT GBDT 模型已成功保存至: {model_path}")
        return model

if __name__ == '__main__':
    trainer = StrictPITLGBTrainer()
    df_matrix = trainer.build_feature_matrix()
    if not df_matrix.empty:
        trainer.train_model(df_matrix)
