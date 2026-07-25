# -*- coding: utf-8 -*-

"""
路径A：大盘择时开关与 IM (中证1000) 股指空头 Beta 对冲引擎
Market Regime Timing Switch & IM Short Beta Hedging Engine
"""

import numpy as np
import pandas as pd

class CBMarketRegimeEngine:
    @staticmethod
    def compute_market_regime(df_panel):
        """
        计算中证转债指数/大盘 5日与 20日移动平均线择时开关
        """
        df_mkt = df_panel.groupby('trade_time')['close'].mean().reset_index()
        df_mkt.rename(columns={'close': 'market_close'}, inplace=True)
        df_mkt['date'] = pd.to_datetime(df_mkt['trade_time']).dt.date
        
        # 每日收盘均线
        daily_mkt = df_mkt.groupby('date')['market_close'].last().reset_index()
        daily_mkt['ma5'] = daily_mkt['market_close'].rolling(5, min_periods=1).mean()
        daily_mkt['ma20'] = daily_mkt['market_close'].rolling(20, min_periods=1).mean()
        
        # 择时信号：MA5 >= MA20 为看多牛市体制；MA5 < MA20 为看空熊市体制
        daily_mkt['is_bullish'] = daily_mkt['ma5'] >= daily_mkt['ma20']
        
        mkt_dict = daily_mkt.set_index('date')['is_bullish'].to_dict()
        mkt_close_dict = daily_mkt.set_index('date')['market_close'].to_dict()
        
        return mkt_dict, mkt_close_dict

    @staticmethod
    def apply_hedging_returns(df_equity, mkt_close_dict, beta=0.70):
        """
        在看空熊市体制下，叠加 IM (中证1000) 股指空头 Beta 对冲
        """
        df_eq = df_equity.copy()
        df_eq['date'] = pd.to_datetime(df_eq['trade_time']).dt.date
        
        daily = df_eq.groupby('date').agg({'nav': 'last'}).reset_index()
        daily['port_ret'] = daily['nav'].pct_change().fillna(0.0)
        daily['mkt_close'] = daily['date'].map(mkt_close_dict).fillna(method='ffill')
        daily['mkt_ret'] = daily['mkt_close'].pct_change().fillna(0.0)
        
        daily['ma5'] = daily['mkt_close'].rolling(5, min_periods=1).mean()
        daily['ma20'] = daily['mkt_close'].rolling(20, min_periods=1).mean()
        daily['is_bearish'] = daily['ma5'] < daily['ma20']
        
        # 择时避险规则：当处于 MA5 < MA20 熊市死叉体制时，坚决 100% 现金空仓避险！
        # 在 MA5 >= MA20 看多体制下，持仓并享受超额；若看空则对冲/空仓
        daily['hedged_ret'] = np.where(
            daily['is_bearish'],
            -beta * daily['mkt_ret'], # IM 股指空头完全做空对冲，捕捉市场下行纯Alpha
            daily['port_ret']
        )
        
        daily['hedged_nav'] = 1000000.0 * (1.0 + daily['hedged_ret']).cumprod()
        return daily
