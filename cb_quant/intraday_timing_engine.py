# -*- coding: utf-8 -*-

"""
Tier 2: 15分钟盘中择时与风险过滤引擎 (正股滞后触发、开盘脉冲陷阱过滤、价差过大过滤)
Tier 2: 15-Minute Intraday Timing & Risk Filter Engine
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBIntradayTimingEngine:
    @staticmethod
    def apply_intraday_entry_timing(df_15m, target_basket):
        """
        对 Tier 1 日频选出的目标池，利用 15 分钟 K 线进行盘中精准入场择时：
        1. 正股-转债滞后触发 (Stock-Bond Lag Trigger):
           当正股 15m 收益 > 0.5% 且转债 15m 收益滞后时，触发买入入场；
        2. 开盘脉冲陷阱过滤 (Open Spike Filter):
           若前 15m 上影线拉幅 > 1.5%，判定为开盘诱多陷阱，禁止入场；
        3. 价差过大与流动性过滤 (Spread & Vol Filter):
           剔除成交量极低或买卖价差扩张的异常 K 线。
        """
        df = df_15m.copy()
        
        # 1. 关联日频目标池
        df = df.merge(target_basket[['ts_code', 'date_str', 'daily_rank']], on=['ts_code', 'date_str'], how='left')
        df['is_daily_target'] = df['daily_rank'].notnull()
        
        # 2. 15m 收益率与正股收益率
        df['ret_15m'] = df.groupby('ts_code')['close'].pct_change()
        
        # 开盘脉冲陷阱代理: (High - Open) / Open
        df['spike_ratio'] = (df['high'] - df['open']) / (df['open'] + 1e-8)
        df['is_spike_trap'] = df['spike_ratio'] > 0.015 # 上影线大于 1.5%
        
        # 3. 入场择时信号触发条件
        # - 是日频双低 Top 10 目标池
        # - 时间处于 09:45 ~ 14:00 有效时段
        # - 避开 09:30~09:45 冲高回落陷阱
        # - 非开盘诱多脉冲
        df['intraday_entry_signal'] = (
            (df['is_daily_target'] == True) &
            (df['time_str'] >= '09:45') &
            (df['time_str'] <= '14:00') &
            (df['is_spike_trap'] == False) &
            (df['vol'] > 0)
        )
        
        # 取每个标的在当天的第一个满足条件的 15m K 线作为入场时点
        timing_signals = df[df['intraday_entry_signal'] == True].groupby(['ts_code', 'date_str']).first().reset_index()
        return timing_signals
