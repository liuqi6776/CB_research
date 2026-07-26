# -*- coding: utf-8 -*-

"""
日期化 PIT 元数据引擎 (As-Of Join 强赎日期、退市日期、历史转股价与正股报价)
Date-Indexed Point-in-Time Metadata Engine (As-Of Join for Redemption, Delisting, & Conversion Price)
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBPITMetadataEngine:
    def __init__(self, mins_data_dir=r"D:\CB_mins_data", data_v2_dir=r"D:\iquant_data\data_v2"):
        self.mins_data_dir = mins_data_dir
        self.data_v2_dir = data_v2_dir

    def apply_pit_asof_join(self, df_15m):
        """
        核心规则：按 (ts_code, trade_date) 维度执行严格的 As-Of Join！
        1. 接入历史强赎公告与行权日期表 (cb_call_history.csv)；
        2. 接入退市日期与发行规模；
        3. 接入 D:\iquant_data\data_v2 真实 T-1 正股收盘价；
        4. 任意日期当 trade_date >= call_date 或 trade_date >= delist_date 时，强制 is_redeemed = True, is_tradable = False！
        绝对禁止任何未来元数据放行！
        """
        df = df_15m.copy()
        df['date_str'] = df['trade_time'].dt.strftime('%Y%m%d')
        df['date_int'] = df['date_str'].astype(int)

        # 1. 接入基础元数据
        basic_path = os.path.join(self.mins_data_dir, "cb_basic_info.csv")
        if os.path.exists(basic_path):
            basic_info = pd.read_csv(basic_path)
            cols = ['ts_code']
            for c in ['stk_code', 'issue_size', 'list_date', 'delist_date', 'conv_price', 'first_conv_price']:
                if c in basic_info.columns:
                    cols.append(c)
            df = df.merge(basic_info[cols], on='ts_code', how='left')

        # 2. 接入强赎日期表 (cb_call_history.csv)
        call_path = os.path.join(self.mins_data_dir, "cb_call_history.csv")
        if os.path.exists(call_path):
            call_info = pd.read_csv(call_path)
            call_info = call_info.dropna(subset=['call_date']).copy()
            call_info['call_date_clean'] = pd.to_numeric(call_info['call_date'], errors='coerce').fillna(20991231)
            df = df.merge(call_info[['ts_code', 'call_date_clean']], on='ts_code', how='left')
        else:
            df['call_date_clean'] = 20991231

        if 'delist_date' in df.columns:
            df['delist_date_clean'] = pd.to_numeric(df['delist_date'], errors='coerce').fillna(20991231)
        else:
            df['delist_date_clean'] = 20991231
            
        df['call_date_clean'] = df['call_date_clean'].fillna(20991231)

        # 3. 强规强赎与退市判定 (As-Of Join)
        df['is_redeemed'] = (df['date_int'] >= df['delist_date_clean']) | (df['date_int'] >= df['call_date_clean'])

        # 4. 接入 D:\iquant_data\data_v2 真实 T-1 正股日线
        day_files = sorted(glob.glob(os.path.join(self.data_v2_dir, "data_day1", "*.parquet")))
        if day_files and 'stk_code' in df.columns:
            day_dfs = []
            for f in day_files:
                try:
                    df_sub = pd.read_parquet(f, columns=['ts_code', 'trade_date', 'close'])
                    day_dfs.append(df_sub)
                except Exception:
                    pass
            if day_dfs:
                stk_daily = pd.concat(day_dfs, ignore_index=True)
                stk_daily.rename(columns={'ts_code': 'stk_code', 'close': 'stk_close_t1'}, inplace=True)
                stk_daily['trade_date_str'] = stk_daily['trade_date'].astype(str)
                stk_daily = stk_daily.sort_values(by=['stk_code', 'trade_date_str']).reset_index(drop=True)
                stk_daily['t1_date_str'] = stk_daily.groupby('stk_code')['trade_date_str'].shift(-1)
                
                df = df.merge(stk_daily[['stk_code', 't1_date_str', 'stk_close_t1']], 
                              left_on=['stk_code', 'date_str'], right_on=['stk_code', 't1_date_str'], how='left')

        # 5. 严格零缺省放行可交易门槛
        raw_amt = pd.to_numeric(df.get('issue_size', np.nan), errors='coerce')
        df['curr_iss_amt'] = np.where(raw_amt > 10000.0, raw_amt / 1e8, raw_amt)
        has_full_pit_metadata = df['stk_close_t1'].notnull() & df['curr_iss_amt'].notnull() & (df['curr_iss_amt'] > 0)

        df['is_tradable'] = (
            has_full_pit_metadata &
            (df['close'] <= 180.0) &
            (df['curr_iss_amt'] >= 2.0) &
            (df['is_redeemed'] == False) &
            (df.get('is_valid_window', True) == True)
        )

        return df
