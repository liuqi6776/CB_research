# -*- coding: utf-8 -*-

"""
Tier 1: 低换手日频可转债因子选债引擎 (双低、纯正股动量、PIT基本面规模)
Tier 1: Low-Turnover Daily Convertible Bond Factor Selection Engine
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBDailyFactorEngine:
    def __init__(self, mins_data_dir=r"D:\CB_mins_data", data_v2_dir=r"D:\iquant_data\data_v2"):
        self.mins_data_dir = mins_data_dir
        self.data_v2_dir = data_v2_dir

    def compute_daily_selection_panel(self, df_15m):
        """
        根据日频双低与 PIT 元数据，按天进行选债
        - 双低因子: 债券价格 + 100 * 转股溢价率
        - 过滤条件: 债券价格 <= 130.0 (防高溢价风险), 剩余规模 >= 2.0 亿, 非退市/赎回
        - 换手率: 低频持仓 3~10 个交易日，极低换手率与极低磨损！
        """
        df = df_15m.copy()
        
        # 按交易日 (date_str) 取每日收盘行
        daily_panel = df.groupby(['ts_code', 'date_str']).last().reset_index()
        
        # 计算双低得分 (Double Low)
        daily_panel['conv_value_t1'] = pd.to_numeric(daily_panel.get('conv_value_t1', np.nan), errors='coerce')
        daily_panel['premium_rate_t1'] = (daily_panel['close'] - daily_panel['conv_value_t1']) / (daily_panel['conv_value_t1'] + 1e-8)
        
        # 如果缺 PIT 正股，退而求其次用纯价格 + 溢价率中枢
        has_prem = daily_panel['premium_rate_t1'].notnull()
        daily_panel['double_low'] = daily_panel['close'] + 100.0 * daily_panel['premium_rate_t1'].fillna(0.30)
        
        # 选债过滤器 (严格低风险防御)
        daily_panel['curr_iss_amt'] = pd.to_numeric(daily_panel.get('issue_size', np.nan), errors='coerce').fillna(5.0)
        daily_panel['delist_date_clean'] = pd.to_numeric(daily_panel.get('delist_date', 20991231), errors='coerce').fillna(20991231)
        daily_panel['date_int'] = daily_panel['date_str'].astype(int)
        daily_panel['is_redeemed'] = daily_panel['date_int'] >= daily_panel['delist_date_clean']
        
        daily_panel['is_daily_eligible'] = (
            (daily_panel['close'] <= 130.0) & # 严格限制低价/中价债，避开高价炒作
            (daily_panel['close'] >= 95.0) &
            (daily_panel['curr_iss_amt'] >= 2.0) &
            (daily_panel['is_redeemed'] == False)
        )
        
        # 每天选出双低得分最低的 Top 10 标的作为目标池
        eligible_df = daily_panel[daily_panel['is_daily_eligible'] == True].copy()
        eligible_df['daily_rank'] = eligible_df.groupby('date_str')['double_low'].rank(ascending=True, method='min')
        
        target_basket = eligible_df[eligible_df['daily_rank'] <= 10][['ts_code', 'date_str', 'double_low', 'daily_rank']]
        return target_basket
