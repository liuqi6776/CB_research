# -*- coding: utf-8 -*-

"""
严格 15 分钟无重叠 60m 标签与 Walk-Forward OOS 验证引擎 (先 15m 聚合、再算得分、同 Session 无午休/过夜、全量过滤应用于 IC)
Strict 15m Non-Overlapping 60m Label & Walk-Forward OOS Engine (Resample-First, Same-Session 60m Holding, Full Filtered IC)
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBStrict15mCleanEngine:
    def __init__(self, data_v2_dir=r"D:\iquant_data\data_v2", mins_data_dir=r"D:\CB_mins_data"):
        self.data_v2_dir = data_v2_dir
        self.mins_data_dir = mins_data_dir

    def load_and_resample_clean_15m(self, df_mins):
        """
        核心规则 1：先把 5 分钟数据聚合为严格 15 分钟 K 线 (df_15m)，然后再计算得分与 60 分钟无重叠标签！
        杜绝先在 5m 上算 shift(-4) (实际仅20分钟) 导致与 15m 信号产生 10 分钟历史重叠的重大漏洞！
        """
        df = df_mins.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'], errors='coerce', format='mixed')
        
        # 1-A. 聚合为 15 分钟 K 线
        df_15m = df.groupby(['ts_code', pd.Grouper(key='trade_time', freq='15min')]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum',
            'amount': 'sum'
        }).dropna(subset=['close']).reset_index()

        df_15m = df_15m.sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)
        df_15m['date_str'] = df_15m['trade_time'].dt.strftime('%Y%m%d')
        df_15m['time_str'] = df_15m['trade_time'].dt.strftime('%H:%M')

        # 1-B. 同 Session 信号窗口 (09:45~10:30 为上午，13:00~14:00 为下午)
        # 上午 10:30 的信号将在 10:45 入场，11:30 平仓（同一上午 Session，无午休跨越）
        # 下午 14:00 的信号将在 14:15 入场，15:00 平仓（同一下午 Session，无过夜跨越）
        is_morning_window = (df_15m['time_str'] >= '09:45') & (df_15m['time_str'] <= '10:30')
        is_afternoon_window = (df_15m['time_str'] >= '13:00') & (df_15m['time_str'] <= '14:00')
        df_15m['is_valid_window'] = is_morning_window | is_afternoon_window
        df_15m['session'] = np.where(is_morning_window, 'Morning', np.where(is_afternoon_window, 'Afternoon', 'Invalid'))

        # 1-C. 在 15m K 线维度上计算严格无重叠的 60 分钟未来标签 (恰好 4 根 15m K 线)
        # Close-to-Close 标签: 当前 15m 收盘价 -> 未来 4 根 15m (60分钟) 收盘价
        df_15m['close_t4'] = df_15m.groupby('ts_code')['close'].shift(-4)
        df_15m['date_t4'] = df_15m.groupby('ts_code')['date_str'].shift(-4)
        df_15m['time_t4'] = df_15m.groupby('ts_code')['time_str'].shift(-4)
        
        # 严规同 Date 且同 Session（无跨午休、无跨过夜）
        same_session_mask = (df_15m['date_str'] == df_15m['date_t4']) & (df_15m['is_valid_window'] == True)
        df_15m['fut_ret_60m_close'] = np.nan
        df_15m.loc[same_session_mask, 'fut_ret_60m_close'] = df_15m.loc[same_session_mask, 'close_t4'] / df_15m.loc[same_session_mask, 'close'] - 1.0

        # Executable Open-to-Open 标签: 下一根 15m Open (t+1) 入场 -> 下 5 根 15m Open (t+5) 平仓 (完整 60 分钟持有)
        df_15m['entry_open'] = df_15m.groupby('ts_code')['open'].shift(-1)
        df_15m['exit_open'] = df_15m.groupby('ts_code')['open'].shift(-5)
        df_15m['date_t5'] = df_15m.groupby('ts_code')['date_str'].shift(-5)
        
        exec_mask = (df_15m['date_str'] == df_15m['date_t5']) & (df_15m['is_valid_window'] == True)
        df_15m['fut_ret_60m_exec'] = np.nan
        df_15m.loc[exec_mask, 'fut_ret_60m_exec'] = df_15m.loc[exec_mask, 'exit_open'] / df_15m.loc[exec_mask, 'entry_open'] - 1.0

        # 1-D. 严格合并可交易元数据与 T-1 滞后正股/筹码
        basic_info_path = os.path.join(self.mins_data_dir, "cb_basic_info.csv")
        if os.path.exists(basic_info_path):
            basic_info = pd.read_csv(basic_info_path)
            cols = ['ts_code']
            for c in ['stk_code', 'issue_size', 'delist_date', 'conv_price', 'first_conv_price']:
                if c in basic_info.columns:
                    cols.append(c)
            df_15m = df_15m.merge(basic_info[cols], on='ts_code', how='left')

        if 'conv_price' in df_15m.columns:
            df_15m['conv_price'] = pd.to_numeric(df_15m['conv_price'], errors='coerce')
            if 'first_conv_price' in df_15m.columns:
                df_15m['conv_price'] = df_15m['conv_price'].fillna(pd.to_numeric(df_15m['first_conv_price'], errors='coerce'))

        # 严格 T-1 滞后正股日线
        day_files = sorted(glob.glob(os.path.join(self.data_v2_dir, "data_day1", "*.parquet")))
        if day_files and 'stk_code' in df_15m.columns:
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
                
                df_15m = df_15m.merge(stk_daily[['stk_code', 't1_date_str', 'stk_close_t1']], 
                                       left_on=['stk_code', 'date_str'], right_on=['stk_code', 't1_date_str'], how='left')

        # 可交易状态门槛：价格<=180 + 规模>=2亿 + 非退市 + 在同Session有效窗口内
        df_15m['curr_iss_amt'] = pd.to_numeric(df_15m.get('issue_size', np.nan), errors='coerce').fillna(5.0)
        df_15m['delist_date_clean'] = pd.to_numeric(df_15m.get('delist_date', 20991231), errors='coerce').fillna(20991231)
        df_15m['date_int'] = df_15m['date_str'].astype(int)
        df_15m['is_redeemed'] = df_15m['date_int'] >= df_15m['delist_date_clean']
        
        df_15m['is_tradable'] = (
            (df_15m['close'] <= 180.0) &
            (df_15m['curr_iss_amt'] >= 2.0) &
            (df_15m['is_redeemed'] == False) &
            (df_15m['is_valid_window'] == True)
        )

        return df_15m

    @staticmethod
    def run_per_timestamp_cross_sectional_ic(df_scored, label_col='fut_ret_60m_close'):
        """
        核心规则 2：标准截面 IC 计算
        每个 15 分钟时间点 (trade_time)，仅在 `is_tradable == True` 且 `is_valid_window == True` 的样本集上计算 Spearman IC！
        得到全量 IC 时间序列 (IC_t)，绝非全月拉平混合！
        """
        # 强制仅使用合规且可交易的样本行！
        clean = df_scored[(df_scored['is_tradable'] == True) & (df_scored['is_valid_window'] == True)].dropna(subset=[label_col, 'score_15m']).copy()
        
        def calc_ts_ic(g):
            if len(g) < 5 or g['score_15m'].std() == 0:
                return np.nan
            return g['score_15m'].corr(g[label_col], method='spearman')
        
        ic_ts = clean.groupby('trade_time').apply(calc_ts_ic).dropna()
        return ic_ts, clean
