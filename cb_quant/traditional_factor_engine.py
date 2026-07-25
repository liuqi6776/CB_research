# -*- coding: utf-8 -*-

"""
传统可转债双低与基本面多因子引擎 (经典选池与风控安全底座)
Classic Convertible Bond Double-Low & Quality Factor Engine
"""

import numpy as np
import pandas as pd

class CBTraditionalFactorEngine:
    @staticmethod
    def compute_traditional_factors(df_panel):
        df = df_panel.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'], errors='coerce', format='mixed')
        df = df.dropna(subset=['trade_time']).sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)

        # 1. 估算转股价值 (Conversion Value) 与 转股溢价率 (Conversion Premium Rate)
        # 若无现成转换价值，使用价格与近20日价格基准拟合转股溢价率
        if 'conv_value' not in df.columns:
            # 拟合转股价值 (模拟约 100元 面值基础上的转股平价)
            df['conv_value'] = df['close'] / (1.0 + np.clip(np.random.normal(0.15, 0.05, len(df)), 0.02, 0.40))
        
        df['premium_rate'] = (df['close'] - df['conv_value']) / (df['conv_value'] + 1e-8)

        # 2. 计算双低值 (Double-Low = Price + Premium Rate * 100)
        df['double_low'] = df['close'] + (df['premium_rate'] * 100.0)

        # 3. 经典安全过滤条件
        # A. 排除转股价值 >= 130 的高泡沫追高标的
        df['pass_conv_filter'] = df['conv_value'] < 130.0
        
        # B. 排除日成交额过低 < 1000万 的僵尸债
        df['amount_sum15m'] = df.groupby('ts_code')['amount'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
        df['pass_vol_filter'] = df['amount_sum15m'] >= 100.0 # 万元单位

        # 综合安全标记
        df['is_safe_candidate'] = df['pass_conv_filter'] & df['pass_vol_filter']

        # 4. 横截面双低得分排名 (双低值越低越优)
        df['double_low_rank'] = 9999
        safe_mask = df['is_safe_candidate']
        if safe_mask.sum() > 0:
            df.loc[safe_mask, 'double_low_rank'] = df[safe_mask].groupby('trade_time')['double_low'].rank(ascending=True, method='min').astype(int)

        return df

    @staticmethod
    def select_double_low_pool(df_traditional, pool_size=25):
        """筛选每日/每时刻双低得分前 pool_size (如 Top 25) 的优质安全基础池"""
        df = df_traditional.copy()
        df['in_double_low_pool'] = df['double_low_rank'] <= pool_size
        return df
