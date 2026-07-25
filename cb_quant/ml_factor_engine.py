# -*- coding: utf-8 -*-

"""
滑点感知与日内微观高频多因子特征工程：包含扣除 0.21% 交易摩擦后的净收益 Label
Friction-Aware ML Factor Engine (Net Alpha Target Label & Microstructure Factors)
"""

import numpy as np
import pandas as pd

class CBMLFactorEngine:
    @staticmethod
    def compute_ml_features(df_panel, friction_cost=0.0021):
        """
        friction_cost: 单边 0.1% 滑点 + 0.005% 佣金 -> 双边交易摩擦共 0.21%
        """
        df = df_panel.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'], errors='coerce', format='mixed')
        df = df.dropna(subset=['trade_time']).sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)
        
        # 1. 日内时空特征
        df['hour'] = df['trade_time'].dt.hour
        df['minute'] = df['trade_time'].dt.minute
        df['minute_of_day'] = df['hour'] * 60 + df['minute']
        df['is_morning'] = (df['hour'] < 12).astype(int)
        
        df['bars_from_open'] = np.where(
            df['is_morning'] == 1,
            (df['minute_of_day'] - 570) // 5,
            (df['minute_of_day'] - 780) // 5 + 24
        )

        grouped = df.groupby('ts_code')

        # 2. 个股多窗口动量、振幅与均值回归反转特征
        df['close_lag1'] = grouped['close'].shift(1)
        df['close_lag3'] = grouped['close'].shift(3)
        df['close_lag6'] = grouped['close'].shift(6)
        df['close_lag12'] = grouped['close'].shift(12)

        df['ret_5m'] = (df['close'] - df['close_lag1']) / (df['close_lag1'] + 1e-8)
        df['ret_15m'] = (df['close'] - df['close_lag3']) / (df['close_lag3'] + 1e-8)
        df['ret_30m'] = (df['close'] - df['close_lag6']) / (df['close_lag6'] + 1e-8)
        df['ret_60m'] = (df['close'] - df['close_lag12']) / (df['close_lag12'] + 1e-8)

        # 反转回调深度 (Mean-Reversion Dip Depth)
        df['high_max15m'] = grouped['high'].rolling(window=3, min_periods=3).max().reset_index(level=0, drop=True)
        df['low_min15m'] = grouped['low'].rolling(window=3, min_periods=3).min().reset_index(level=0, drop=True)
        df['amp_15m'] = (df['high_max15m'] - df['low_min15m']) / (df['close_lag3'] + 1e-8)
        df['dip_depth_15m'] = (df['close'] - df['high_max15m']) / (df['high_max15m'] + 1e-8)

        # 3. 量能突变倍率 (Volume Surge)
        df['amount_sum15m'] = grouped['amount'].rolling(window=3, min_periods=3).sum().reset_index(level=0, drop=True)
        df['amount_ma5d'] = grouped['amount_sum15m'].rolling(window=240, min_periods=20).mean().reset_index(level=0, drop=True)
        df['vol_surge_15m'] = df['amount_sum15m'] / (df['amount_ma5d'] + 1e-5)

        # ATR 相对真实波幅
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close_lag1']).abs()
        df['tr3'] = (df['low'] - df['close_lag1']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_14'] = grouped['tr'].rolling(window=14, min_periods=5).mean().reset_index(level=0, drop=True)
        df['atr_pct'] = df['atr_14'] / (df['close'] + 1e-8)

        # 4. 市场微观体征特征
        df['market_ret_15m'] = df.groupby('trade_time')['ret_15m'].transform('mean')
        df['market_amp_15m'] = df.groupby('trade_time')['amp_15m'].transform('mean')
        df['advance_decline_ratio'] = df.groupby('trade_time')['ret_15m'].transform(lambda x: (x > 0).mean())

        # 5. ★ 核心创新：扣除 0.21% 双边交易摩擦后的滑点感知净收益 Label
        df['close_future12'] = grouped['close'].shift(-12)
        df['fut_ret_60m'] = (df['close_future12'] - df['close']) / (df['close'] + 1e-8)
        
        market_fut_ret = df.groupby('trade_time')['fut_ret_60m'].transform('mean')
        df['fut_alpha_60m'] = df['fut_ret_60m'] - market_fut_ret
        
        # 扣除交易摩擦成本后的净 Alpha (Net Alpha > +0.3% 对应二分类 1)
        df['fut_alpha_60m_net'] = df['fut_alpha_60m'] - friction_cost
        df['label_upward_alpha'] = (df['fut_alpha_60m_net'] > 0.0030).astype(int)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    @staticmethod
    def get_feature_columns():
        return [
            'ret_5m', 'ret_15m', 'ret_30m', 'ret_60m',
            'amp_15m', 'dip_depth_15m', 'vol_surge_15m', 'atr_pct',
            'minute_of_day', 'is_morning', 'bars_from_open',
            'market_ret_15m', 'market_amp_15m', 'advance_decline_ratio'
        ]
