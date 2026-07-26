# -*- coding: utf-8 -*-

"""
收益率极大值幅度因子引擎 (Extreme Return Magnitude / ex_rtn_max_val Factor Engine)
参考研报: 兴业证券 郑兆磊《高频研究系列三：收益率分布中的Alpha（2）》
算法逻辑:
1. 1分钟收益率 cbond_rtn1 与 5分钟滚动收益率 cbond_rtn5
2. 单日 1分钟收益率 5% 与 95% 分位数 (VaR 门槛 q_05, q_95)
3. ex_rtn_max_val_5min = (mean(cbond_rtn5 > q_95) * cbond_rtn5) / q_95
4. ex_rtn_min_freq_5min = sum(cbond_rtn5 < q_05)
5. ex_rtn_max_val_1min = (mean(cbond_rtn1 > q_95) * cbond_rtn1) / q_95
6. ex_rtn_min_freq_1min = sum(cbond_rtn1 < q_05)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class CBExtremeReturnFactorEngine:
    """
    收益率极大值幅度 (Extreme Return Magnitude) 因子计算器
    """
    def __init__(self):
        pass

    def compute_daily_factor(self, df_1m_d):
        """
        计算单日单标的的 4 种收益率极大值/极小值因子
        输入: df_1m_d (包含 close 序列，或者按 15m 充当近似分钟序列)
        """
        # 1. 1分钟收益率与 5分钟(或5周期) 收益率
        rtn1 = df_1m_d['close'].pct_change()
        rtn5 = df_1m_d['close'].pct_change(5)
        
        # 2. VaR 95% 与 5% 门槛
        q_05 = rtn1.quantile(0.05)
        q_95 = rtn1.quantile(0.95)
        
        if np.isnan(q_95) or q_95 <= 0:
            return pd.Series({
                'ex_rtn_max_val_5min': np.nan, 'ex_rtn_min_freq_5min': np.nan,
                'ex_rtn_max_val_1min': np.nan, 'ex_rtn_min_freq_1min': np.nan
            })
            
        # 3. 5分钟极大值幅度与极小值频率
        r5_tail = rtn5[rtn5 > q_95]
        ex_max_5m = (r5_tail.mean() / q_95) if len(r5_tail) > 0 else 1.0
        ex_min_f_5m = float((rtn5 < q_05).sum()) if not np.isnan(q_05) else 0.0
        
        # 4. 1分钟极大值幅度与极小值频率
        r1_tail = rtn1[rtn1 > q_95]
        ex_max_1m = (r1_tail.mean() / q_95) if len(r1_tail) > 0 else 1.0
        ex_min_f_1m = float((rtn1 < q_05).sum()) if not np.isnan(q_05) else 0.0
        
        return pd.Series({
            'ex_rtn_max_val_5min': ex_max_5m,
            'ex_rtn_min_freq_5min': ex_min_f_5m,
            'ex_rtn_max_val_1min': ex_max_1m,
            'ex_rtn_min_freq_1min': ex_min_f_1m
        })

    def generate_extreme_return_panel(self, df_panel):
        logging.info("=== 开始构建收益率极大值幅度 (ex_rtn_max_val) 超高速矢量化因子面板 ===")
        df_panel['date_str'] = pd.to_datetime(df_panel['trade_time']).dt.strftime('%Y%m%d')
        
        # 1. 组内计算 15m 收益率与 5-bar (75m) 收益率
        df_panel['rtn1'] = df_panel.groupby(['date_str', 'ts_code'])['close'].pct_change()
        df_panel['rtn5'] = df_panel.groupby(['date_str', 'ts_code'])['close'].pct_change(5)
        
        # 2. 每日标的算 95% 与 5% quantile 门槛
        q_df = df_panel.groupby(['date_str', 'ts_code'])['rtn1'].quantile([0.05, 0.95]).unstack()
        q_df.columns = ['q05', 'q95']
        
        df_panel = df_panel.merge(q_df, on=['date_str', 'ts_code'], how='left')
        
        # 3. 极大值幅度
        df_panel['r5_tail'] = np.where(df_panel['rtn5'] > df_panel['q95'], df_panel['rtn5'], np.nan)
        df_panel['r1_tail'] = np.where(df_panel['rtn1'] > df_panel['q95'], df_panel['rtn1'], np.nan)
        df_panel['r5_min_flag'] = np.where(df_panel['rtn5'] < df_panel['q05'], 1.0, 0.0)
        df_panel['r1_min_flag'] = np.where(df_panel['rtn1'] < df_panel['q05'], 1.0, 0.0)
        
        agg_df = df_panel.groupby(['date_str', 'ts_code']).agg({
            'r5_tail': 'mean',
            'r1_tail': 'mean',
            'q95': 'last',
            'r5_min_flag': 'sum',
            'r1_min_flag': 'sum'
        }).reset_index()
        
        agg_df['ex_rtn_max_val_5min'] = agg_df['r5_tail'] / (agg_df['q95'] + 1e-8)
        agg_df['ex_rtn_max_val_1min'] = agg_df['r1_tail'] / (agg_df['q95'] + 1e-8)
        agg_df['ex_rtn_min_freq_5min'] = agg_df['r5_min_flag']
        agg_df['ex_rtn_min_freq_1min'] = agg_df['r1_min_flag']
        
        # 填充默认值
        agg_df['ex_rtn_max_val_5min'] = agg_df['ex_rtn_max_val_5min'].fillna(1.0)
        agg_df['ex_rtn_max_val_1min'] = agg_df['ex_rtn_max_val_1min'].fillna(1.0)
        
        logging.info(f"超高速矢量化收益率极大值幅度因子构建完成，共生成 {len(agg_df):,} 条记录。")
        return agg_df

if __name__ == '__main__':
    loader = CBDataLoader()
    df_p = loader.load_minute_panel(start_date="2025-01-01", end_date="2025-02-01", max_bonds=100)
    engine = CBExtremeReturnFactorEngine()
    df_f = engine.generate_extreme_return_panel(df_p)
    print(df_f.head(10))
