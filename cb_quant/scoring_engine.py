# -*- coding: utf-8 -*-

"""
可转债横截面打分与选债引擎
Convertible Bond Cross-Sectional Ranking & Scoring Engine
"""

import pandas as pd
import numpy as np

class CBScoringEngine:
    def __init__(self, weights=None, top_n=8, min_amount=500000.0):
        """
        初始化打分引擎
        weights: 因子权重配置
        """
        if weights is None:
            self.weights = {
                'smooth_momentum_15m': 0.30, # 路径平滑动量 30%
                'vol_amplitude_30m': 0.20,    # 30分钟振幅 20%
                'volume_ratio_15m': 0.20,     # 相对成交量比率 20%
                'relative_alpha_15m': 0.30    # 相对大盘超额 30%
            }
        else:
            self.weights = weights
            
        self.top_n = top_n
        self.min_amount = min_amount

    def compute_cross_sectional_scores(self, df_factors):
        """
        按时间点 (trade_time) 进行横截面百分位排名打分 (Percentile Ranking)
        """
        df = df_factors.copy()
        
        # 1. 过滤成交活跃度门槛
        df['is_valid'] = df['amount_sum15m'] >= self.min_amount
        
        # 过滤包含无效因子的记录
        factor_cols = list(self.weights.keys())
        df['has_nan'] = df[factor_cols].isna().any(axis=1)
        valid_mask = df['is_valid'] & (~df['has_nan'])
        
        df_valid = df[valid_mask].copy()
        
        if df_valid.empty:
            df['total_score'] = 0.0
            df['rank'] = 9999
            return df
            
        # 2. 按 trade_time 进行横截面 Percentile Rank (0~1)
        for factor in factor_cols:
            rank_col = f'{factor}_rank'
            df_valid[rank_col] = df_valid.groupby('trade_time')[factor].rank(pct=True, ascending=True)

        # 3. 计算加权综合得分
        df_valid['total_score'] = 0.0
        for factor, weight in self.weights.items():
            rank_col = f'{factor}_rank'
            df_valid['total_score'] += df_valid[rank_col] * weight

        # 4. 横截面排名排序
        df_valid['rank'] = df_valid.groupby('trade_time')['total_score'].rank(pct=False, ascending=False, method='min')

        # 合并结果回主表
        df = df.merge(df_valid[['ts_code', 'trade_time', 'total_score', 'rank']], 
                      on=['ts_code', 'trade_time'], 
                      how='left')
        
        df['total_score'] = df['total_score'].fillna(0.0)
        df['rank'] = df['rank'].fillna(9999).astype(int)
        return df
