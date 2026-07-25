# -*- coding: utf-8 -*-

"""
LightGBM 二分类置信度模型与排名概率预测引擎
LightGBM Classification & Confidence Rank Prediction Engine
"""

import sys
import logging
import numpy as np
import pandas as pd

# Bypass outdated dask accessor compatibility issue
sys.modules['dask'] = None
sys.modules['dask.dataframe'] = None
import lightgbm as lgb
from cb_quant.ml_factor_engine import CBMLFactorEngine

logger = logging.getLogger(__name__)

class CBMLModelEngine:
    def __init__(self, n_estimators=120, learning_rate=0.03, max_depth=5, num_leaves=31, min_confidence=0.55):
        """
        min_confidence: 置信度过滤门槛 (概率 >= 0.55 属于高胜率上行标的)
        """
        self.feature_cols = CBMLFactorEngine.get_feature_columns()
        self.min_confidence = min_confidence
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.is_trained = False

    def train(self, df_train):
        """
        训练 LightGBM 上行二分类模型 (预测 Alpha 60m > +0.3% 概率)
        """
        df_clean = df_train.dropna(subset=self.feature_cols + ['label_upward_alpha']).copy()
        X = df_clean[self.feature_cols]
        y = df_clean['label_upward_alpha']
        
        pos_ratio = (y == 1).mean()
        logger.info(f"开始训练 LightGBM 上行二分类模型 (训练集: {len(X):,} 条, 上行正样本占比: {pos_ratio*100:.2f}%)...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("LightGBM 置信度模型训练完成！")

    def predict_ranks(self, df_factors):
        """
        输入因子面板数据，生成概率置信度得分与置信度过滤后的横截面排名
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 train() 函数。")
            
        df = df_factors.copy()
        valid_mask = ~df[self.feature_cols].isna().any(axis=1)
        
        df['prob_upward'] = 0.0
        if valid_mask.sum() > 0:
            probs = self.model.predict_proba(df.loc[valid_mask, self.feature_cols])[:, 1]
            df.loc[valid_mask, 'prob_upward'] = probs
            
        # 攻坚点1：置信度过滤 (概率未达 0.55 标的赋 0，不参与排名)
        df['is_high_confidence'] = df['prob_upward'] >= self.min_confidence
        df['filtered_score'] = np.where(df['is_high_confidence'], df['prob_upward'], 0.0)
        
        # 横截面按置信度得分降序排名
        df['rank'] = df.groupby('trade_time')['filtered_score'].rank(ascending=False, method='min').fillna(9999).astype(int)
        # 如果得分全为 0 (全市场无高置信度标的)，将 rank 重置为 9999 (空仓观望)
        df['rank'] = np.where(df['filtered_score'] > 0.0, df['rank'], 9999)
        return df
