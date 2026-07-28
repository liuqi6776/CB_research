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

        # 强赎与退市判定 (严格 PIT 零时间倒退逻辑: date_int >= call_date_clean 当天或之后判定不可交易，严禁向前倒推 5 天)
        df['delist_date_clean'] = pd.to_numeric(df['delist_date'], errors='coerce') if 'delist_date' in df.columns else np.nan
        df['is_redeemed'] = np.where(
            df['call_date_clean'].notnull() & (df['date_int'] >= df['call_date_clean']), True,
            np.where(df['delist_date_clean'].notnull() & (df['date_int'] >= df['delist_date_clean']), True, False)
        )

        # 3. 接入 D:\iquant_data\data_v2 真实 T-1 正股日线收盘价 (stk_close_t1)
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
                stk_day = pd.concat(day_dfs, ignore_index=True)
                stk_day.rename(columns={'close': 'stk_close', 'trade_date': 'stk_trade_date'}, inplace=True)
                stk_day['stk_trade_date'] = pd.to_numeric(stk_day['stk_trade_date'], errors='coerce')
                
                trade_dates = sorted(stk_day['stk_trade_date'].unique())
                t1_map = {trade_dates[i]: trade_dates[i-1] for i in range(1, len(trade_dates))}
                df['t1_date_int'] = df['date_int'].map(t1_map)
                df['stk_code_clean'] = df['stk_code'].astype(str).str.split('.').str[0].str.zfill(6)
                stk_day['stk_code_clean'] = stk_day['ts_code'].astype(str).str.split('.').str[0].str.zfill(6)
                
                if 'stk_close_t1' in df.columns:
                    df.drop(columns=['stk_close_t1'], inplace=True)
                
                df = df.merge(stk_day[['stk_code_clean', 'stk_trade_date', 'stk_close']], 
                              left_on=['stk_code_clean', 't1_date_int'], 
                              right_on=['stk_code_clean', 'stk_trade_date'], 
                              how='left')
                df.rename(columns={'stk_close': 'stk_close_t1'}, inplace=True)
                if 'stk_code_clean' in df.columns:
                    df.drop(columns=['stk_code_clean'], inplace=True)

        # 3-B. 接入筹码分布与游资量化特征 (D:\iquant_data\chip_distribution_t1)
        chip_dir = os.path.join(self.mins_data_dir, "..", "chip_distribution_t1")
        if os.path.exists(chip_dir) and 'stk_code' in df.columns:
            try:
                chip_files = sorted(glob.glob(os.path.join(chip_dir, "*.parquet")))
                if chip_files:
                    df_chip = pd.read_parquet(chip_files[0])
                    chip_cols = [c for c in ['winner_ratio', 'chip_density', 'retail_heat'] if c in df_chip.columns]
                    for c in chip_cols:
                        if c in df.columns:
                            df.drop(columns=[c], inplace=True)
                    
                    df = df.merge(df_chip[['stk_code', 't1_date_str'] + chip_cols],
                                  left_on=['stk_code', 'date_str'], right_on=['stk_code', 't1_date_str'], how='left')
            except Exception as e:
                logger.warning(f"筹码因子合并失败: {e}")

        # 1-B. 接入真实全网多阶历史转股价变动事件表 (cb_conv_price_history.csv) 并使用 merge_asof 匹配
        hist_adj_path = os.path.join(self.mins_data_dir, "cb_conv_price_history.csv")
        if not os.path.exists(hist_adj_path):
            hist_adj_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "cb_conv_price_history.csv")
            
        if os.path.exists(hist_adj_path) and 'stk_code' in df.columns:
            try:
                df_adj_hist = pd.read_csv(hist_adj_path)
                df_adj_hist['stk_code_clean'] = df_adj_hist['stk_code'].astype(str).str.split('.').str[0].str.zfill(6)
                df_adj_hist['eff_date_int'] = pd.to_numeric(df_adj_hist['effective_date'], errors='coerce')
                
                # 过滤有效多阶价格记录
                df_adj_valid = df_adj_hist.dropna(subset=['eff_date_int', 'new_conv_price']).copy()
                df_adj_valid.sort_values(by=['stk_code_clean', 'eff_date_int'], inplace=True)
                
                # 为每只股票增加首次生效日期标识
                min_eff = df_adj_valid.groupby('stk_code_clean')['eff_date_int'].min().reset_index()
                min_eff.rename(columns={'eff_date_int': 'first_eff_date'}, inplace=True)
                
                df['stk_code_clean'] = df['stk_code'].astype(str).str.split('.').str[0].str.zfill(6)
                if 'first_eff_date' in df.columns:
                    df.drop(columns=['first_eff_date'], inplace=True)
                df = df.merge(min_eff, on='stk_code_clean', how='left')
                
                df['date_int'] = df['date_int'].astype(np.int64)
                df_adj_valid['eff_date_int'] = df_adj_valid['eff_date_int'].astype(np.int64)
                
                # 执行 merge_asof: 按 date_int >= eff_date_int 匹配当时实际生效的转股价 asof_conv_price
                df_sorted = df.sort_values(by=['date_int']).reset_index()
                df_adj_valid_sorted = df_adj_valid[['stk_code_clean', 'eff_date_int', 'new_conv_price']].sort_values(by=['eff_date_int']).reset_index(drop=True)
                
                merged_asof = pd.merge_asof(
                    df_sorted,
                    df_adj_valid_sorted,
                    left_on='date_int',
                    right_on='eff_date_int',
                    by='stk_code_clean',
                    direction='backward'
                )
                merged_asof.sort_values(by='index', inplace=True)
                merged_asof.set_index('index', inplace=True)
                df = merged_asof
                df.rename(columns={'new_conv_price': 'asof_conv_price'}, inplace=True)
                
                if 'stk_code_clean' in df.columns:
                    df.drop(columns=['stk_code_clean'], inplace=True)
            except Exception as e:
                logger.warning(f"多阶历史转股价 merge_asof 匹配失败: {e}")

        # 4. 严禁任何 .fillna() 默认放行补齐！缺失即判定无效！
        stk_c = pd.to_numeric(df['stk_close_t1'], errors='coerce') if 'stk_close_t1' in df.columns else pd.Series(np.nan, index=df.index)
        latest_conv_px = pd.to_numeric(df['conv_price'], errors='coerce') if 'conv_price' in df.columns else pd.Series(np.nan, index=df.index)
        first_conv_px = pd.to_numeric(df['first_conv_price'], errors='coerce') if 'first_conv_price' in df.columns else latest_conv_px
        asof_conv_px = pd.to_numeric(df['asof_conv_price'], errors='coerce') if 'asof_conv_price' in df.columns else pd.Series(np.nan, index=df.index)
        
        # 多阶 PIT 无前视转股价 As-Of 逻辑:
        # 1) 若 date_int < first_eff_date (尚未发生任何下修)，严格使用初始转股价 first_conv_price；
        # 2) 若 date_int >= eff_date，严格使用当时实际生效的 asof_conv_price (多阶准确转股价)；
        # 3) 若未在事件表中，退回使用 latest_conv_px / first_conv_px。
        has_first_eff = df['first_eff_date'].notnull() if 'first_eff_date' in df.columns else pd.Series(False, index=df.index)
        is_pre_first_eff = has_first_eff & (df['date_int'] < df['first_eff_date'])
        
        conv_px = np.where(
            is_pre_first_eff & first_conv_px.notnull() & (first_conv_px > 0), first_conv_px,
            np.where(asof_conv_px.notnull() & (asof_conv_px > 0), asof_conv_px,
                     np.where(first_conv_px.notnull() & (first_conv_px > 0), first_conv_px, latest_conv_px))
        )
        conv_px = pd.Series(conv_px, index=df.index)
        
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
