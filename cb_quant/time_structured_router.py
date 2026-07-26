# -*- coding: utf-8 -*-

"""
严格时间结构路由与订单执行器 (Time-Structured Signal & Execution Router)
确保 feature_date < trade_date 且 signal_time < execution_time
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBTimeStructuredRouter:
    @staticmethod
    def generate_time_structured_orders(df_pit):
        """
        全流程时间结构路由：
        1. T-1 日收盘 (feature_date = T-1) 计算选债得分，生成适用于 T 日 (trade_date = T) 的目标池；
        2. T 盘中 09:45 K 线收盘形成信号 (signal_time = 09:45)；
        3. T 盘中下一根 10:00 K 线开盘成交 (execution_time = 10:00, execution_price = 10:00 Open)；
        4. 严格校验 feature_time < signal_time < execution_time，保留完整时间戳结构。
        """
        df = df_pit.copy()
        
        # A. 提取每日收盘行，按 feature_date (T-1) 选出适合下一交易日 (trade_date) 的目标池
        daily_panel = df.groupby(['ts_code', 'date_str']).last().reset_index()
        daily_panel = daily_panel.sort_values(by=['ts_code', 'date_str']).reset_index(drop=True)
        
        # 显式计算 feature_date (T-1) 与 trade_date (T)
        unique_dates = sorted(daily_panel['date_str'].unique())
        date_map = {unique_dates[i]: unique_dates[i+1] for i in range(len(unique_dates)-1)}
        
        daily_panel['feature_date'] = daily_panel['date_str']
        daily_panel['trade_date'] = daily_panel['feature_date'].map(date_map)
        
        # 仅挑选特性能日在交易日之前 (feature_date < trade_date) 且属于选债资格 (is_eligible_at_selection) 的标的
        eligible_panel = daily_panel[
            daily_panel['trade_date'].notnull() & 
            (daily_panel['is_eligible_at_selection'] == True)
        ].copy()
        
        eligible_panel['daily_rank'] = eligible_panel.groupby('feature_date')['double_low'].rank(ascending=True, method='min')
        target_basket = eligible_panel[eligible_panel['daily_rank'] <= 10][['ts_code', 'feature_date', 'trade_date', 'daily_rank', 'double_low']]
        
        # B. 匹配 T 日 (trade_date) 盘中 15m 信号
        # 将目标池与其在 trade_date 当天的 15m 行情合并
        df_trade = df.merge(target_basket, left_on=['ts_code', 'date_str'], right_on=['ts_code', 'trade_date'], how='inner')
        
        # C. 在 09:45 发起盘中信号
        signal_rows = df_trade[
            (df_trade['is_executable_at_signal'] == True) &
            (df_trade['time_str'] >= '09:45') &
            (df_trade['time_str'] <= '14:00')
        ].copy()
        
        if signal_rows.empty:
            return pd.DataFrame(), target_basket
            
        # 取每个标的在 trade_date 当天的第一个触发信号
        first_signals = signal_rows.groupby(['ts_code', 'trade_date']).first().reset_index()
        
        # D. 匹配下一根 K 线作为成交 Fill
        orders = []
        for idx, sig in first_signals.iterrows():
            code = sig['ts_code']
            t_date = sig['trade_date']
            sig_time = sig['trade_time']
            
            # 在 trade_date 当天找 trade_time > sig_time 的第一根 K 线成交
            df_sub = df[(df['ts_code'] == code) & (df['date_str'] == t_date) & (df['trade_time'] > sig_time)]
            if not df_sub.empty:
                fill_row = df_sub.iloc[0]
                if fill_row.get('is_executable_at_fill', False):
                    orders.append({
                        'ts_code': code,
                        'feature_date': sig['feature_date'],
                        'trade_date': t_date,
                        'signal_time': str(sig_time),
                        'execution_time': str(fill_row['trade_time']),
                        'execution_price': fill_row['open'],
                        'execution_vol': fill_row['vol'],
                        'daily_rank': sig['daily_rank'],
                        'double_low': sig['double_low']
                    })

        df_orders = pd.DataFrame(orders)
        return df_orders, target_basket
