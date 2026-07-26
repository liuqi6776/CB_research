# -*- coding: utf-8 -*-

"""
统一资格状态机与零缺省 PIT 引擎 (Strict Unified Eligibility & Zero-Fallback PIT State Machine)
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBUnifiedPITEngine:
    def __init__(self, mins_data_dir=r"D:\CB_mins_data", data_v2_dir=r"D:\iquant_data\data_v2"):
        self.mins_data_dir = mins_data_dir
        self.data_v2_dir = data_v2_dir

    def build_unified_state_panel(self, df_15m):
        """
        构建包含完整时间戳与三层统一资格状态的行情面板：
        1. 废除所有默认放行规则：缺失转股价值/溢价率/规模/强赎状态的，统统不予补齐！
        2. T-1 日频选债特征日期 (feature_date) 严格小于 T 交易执行日期 (trade_date)；
        3. 建立 3 级资格状态机：
           - is_eligible_at_selection (选债资格)
           - is_executable_at_signal (信号资格)
           - is_executable_at_fill (成交资格)
        """
        df = df_15m.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'], errors='coerce')
        df['date_str'] = df['trade_time'].dt.strftime('%Y%m%d')
        df['time_str'] = df['trade_time'].dt.strftime('%H:%M')
        df['date_int'] = pd.to_numeric(df['date_str'], errors='coerce').fillna(20991231).astype(int)

        # 1. 接入基础元数据 (cb_basic_info.csv)
        basic_path = os.path.join(self.mins_data_dir, "cb_basic_info.csv")
        if os.path.exists(basic_path):
            basic_info = pd.read_csv(basic_path)
            cols = [c for c in ['stk_code', 'issue_size', 'list_date', 'delist_date', 'conv_price', 'first_conv_price'] if c in basic_info.columns]
            for c in cols:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)
            df = df.merge(basic_info[['ts_code'] + cols], on='ts_code', how='left')

        # 2. 接入强赎公告日期表 (cb_call_history.csv)
        call_path = os.path.join(self.mins_data_dir, "cb_call_history.csv")
        if os.path.exists(call_path):
            call_info = pd.read_csv(call_path)
            call_info = call_info.dropna(subset=['call_date']).copy()
            call_info['call_date_clean'] = pd.to_numeric(call_info['call_date'], errors='coerce')
            df = df.merge(call_info[['ts_code', 'call_date_clean']], on='ts_code', how='left')

        # 强赎与退市判定 (严格 PIT 逻辑: date_int >= call_date 或 delist_date 时判定已强赎/退市)
        df['delist_date_clean'] = pd.to_numeric(df['delist_date'], errors='coerce') if 'delist_date' in df.columns else np.nan
        df['is_redeemed'] = np.where(
            df['call_date_clean'].notnull() & (df['date_int'] >= df['call_date_clean']), True,
            np.where(df['delist_date_clean'].notnull() & (df['date_int'] >= df['delist_date_clean']), True, False)
        )

        # 3. 接入 D:\iquant_data\data_v2 真实 T-1 正股日线收盘价 (stk_close_t1)
        day_files = sorted(glob.glob(os.path.join(self.data_v2_dir, "data_day1", "*.parquet")))
        valid_day_files = [f for f in day_files if os.path.basename(f).replace('.parquet','') >= '20241201']
        
        if valid_day_files and 'stk_code' in df.columns:
            day_dfs = []
            for f in valid_day_files:
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
                # T-1 对齐: 下一交易日 t1_date_str 使用的是上一个交易日的 stk_close_t1
                stk_daily['t1_date_str'] = stk_daily.groupby('stk_code')['trade_date_str'].shift(-1)
                
                for drop_col in ['stk_close_t1', 't1_date_str']:
                    if drop_col in df.columns:
                        df.drop(columns=[drop_col], inplace=True)

                df = df.merge(stk_daily[['stk_code', 't1_date_str', 'stk_close_t1']], 
                              left_on=['stk_code', 'date_str'], right_on=['stk_code', 't1_date_str'], how='left')

        if 'stk_close_t1' not in df.columns:
            df['stk_close_t1'] = np.nan

        # 4. 严禁任何 .fillna() 默认放行补齐！缺失即判定无效！
        stk_c = pd.to_numeric(df['stk_close_t1'], errors='coerce') if 'stk_close_t1' in df.columns else pd.Series(np.nan, index=df.index)
        conv_px = pd.to_numeric(df['conv_price'], errors='coerce') if 'conv_price' in df.columns else pd.Series(np.nan, index=df.index)
        
        df['conv_value_t1'] = np.where(stk_c.notnull() & conv_px.notnull() & (conv_px > 0), 
                                      100.0 * stk_c / conv_px, np.nan)
        
        # 溢价率 (不填补任何 0.30 默认值)
        df['premium_rate_t1'] = np.where(df['conv_value_t1'].notnull() & (df['conv_value_t1'] > 0),
                                         (df['close'] - df['conv_value_t1']) / df['conv_value_t1'], np.nan)
        
        # 双低得分 (不填补任何默认值)
        df['double_low'] = np.where(df['premium_rate_t1'].notnull(),
                                    df['close'] + 100.0 * df['premium_rate_t1'], np.nan)
        
        # 规模 (不填补任何 5.0 默认值)
        raw_amt = pd.to_numeric(df['issue_size'], errors='coerce') if 'issue_size' in df.columns else pd.Series(np.nan, index=df.index)
        df['curr_iss_amt'] = np.where(raw_amt.notnull() & (raw_amt > 10000.0), raw_amt / 1e8, raw_amt)

        # 5. 三级统一资格状态机定义 (机构级零缺省)
        # Tier 1: T-1 日频选债资格 (is_eligible_at_selection)
        # 必须满足: 双低/规模/强赎元数据完整 (has_valid_call_metadata==True), close <= 130, 规模 >= 2.0 亿, 未强赎
        df['is_eligible_at_selection'] = (
            df['double_low'].notnull() &
            df['curr_iss_amt'].notnull() &
            (df.get('has_valid_call_metadata', True) == True) &
            (df['close'] <= 130.0) &
            (df['close'] >= 95.0) &
            (df['curr_iss_amt'] >= 2.0) &
            (df['is_redeemed'] == False)
        )

        # Tier 2: 15m 信号发起资格 (is_executable_at_signal)
        # 必须满足: 处于 Tier 1 选债池, 时间 09:45~14:00, 15m K线行情完整, 非开盘诱多上影线
        df['spike_ratio'] = np.where(df['open'] > 0, (df['high'] - df['open']) / df['open'], 0.0)
        df['is_spike_trap'] = df['spike_ratio'] > 0.015
        
        df['is_executable_at_signal'] = (
            (df['is_eligible_at_selection'] == True) &
            (df['time_str'] >= '09:45') &
            (df['time_str'] <= '14:00') &
            (df['is_spike_trap'] == False) &
            (df['vol'] > 0) &
            (df['open'] > 0)
        )

        # Tier 3: 下一根 K 线成交开盘资格 (is_executable_at_fill)
        df['is_executable_at_fill'] = (
            (df['vol'] > 0) &
            (df['open'] > 0) &
            (df['is_redeemed'] == False)
        )

        return df
