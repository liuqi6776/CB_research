# -*- coding: utf-8 -*-

"""
可转债分钟级多因子计算引擎
Convertible Bond Minute Multi-Factor Engine
"""

import numpy as np
import pandas as pd

class CBFactorEngine:
    """
    计算基于 5 分钟 K 线的动量、路径平滑度、相对强弱 Alpha、振幅及成交量突变因子
    """
    
    @staticmethod
    def compute_factors(df_panel):
        """
        输入面板数据，计算全量分钟因子
        df_panel columns: ['ts_code', 'trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount']
        """
        df = df_panel.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'], errors='coerce', format='mixed')
        df = df.sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)
        
        # 1. 基础滑动窗口聚合 (5分钟线 3 根 = 15 分钟，6 根 = 30 分钟)
        # 按股票分组计算 Rolling 指标
        grouped = df.groupby('ts_code')
        
        # 1. 简单 15 分钟收益率 (3 根 5min 棒)
        df['close_lag3'] = grouped['close'].shift(3)
        df['return_15m'] = (df['close'] - df['close_lag3']) / (df['close_lag3'] + 1e-8)
        
        # 2. 15 分钟区间最高价与最低价 -> 计算振幅
        df['high_max15m'] = grouped['high'].rolling(window=3, min_periods=3).max().reset_index(level=0, drop=True)
        df['low_min15m'] = grouped['low'].rolling(window=3, min_periods=3).min().reset_index(level=0, drop=True)
        df['amplitude_15m'] = (df['high_max15m'] - df['low_min15m']) / (df['close_lag3'] + 1e-8)
        
        # 收益率路径平滑度 = 15分钟收益率 / (15分钟振幅 + eps)
        df['smooth_momentum_15m'] = df['return_15m'] / (df['amplitude_15m'] + 1e-5)
        
        # 3. 30 分钟波动振幅 (6 根 5min 棒)
        df['close_lag6'] = grouped['close'].shift(6)
        df['high_max30m'] = grouped['high'].rolling(window=6, min_periods=6).max().reset_index(level=0, drop=True)
        df['low_min30m'] = grouped['low'].rolling(window=6, min_periods=6).min().reset_index(level=0, drop=True)
        df['vol_amplitude_30m'] = (df['high_max30m'] - df['low_min30m']) / (df['close_lag6'] + 1e-8)
        
        # 4. 近 15 分钟成交额之和及相对成交量比率 (Volume Ratio)
        df['amount_sum15m'] = grouped['amount'].rolling(window=3, min_periods=3).sum().reset_index(level=0, drop=True)
        # 近 5 日同频滚动平均成交额 (以 48 根/天 * 5 天 = 240 根为参考基准)
        df['amount_ma5d'] = grouped['amount_sum15m'].rolling(window=240, min_periods=20).mean().reset_index(level=0, drop=True)
        df['volume_ratio_15m'] = df['amount_sum15m'] / (df['amount_ma5d'] + 1e-5)
        
        # 5. 计算全市场同频平均收益率 -> 导出相对强弱 Alpha
        market_ret = df.groupby('trade_time')['return_15m'].transform('mean')
        df['relative_alpha_15m'] = df['return_15m'] - market_ret
        
        # 填充异常值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df
