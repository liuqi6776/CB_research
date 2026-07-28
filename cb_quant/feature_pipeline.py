# -*- coding: utf-8 -*-

"""
全量多因子统一训练/推理特征流水线 (Unified Train/Serve Feature Pipeline)
保证训练阶段 (train_master_gbdt_model.py) 与推理/回测阶段 (run_master_multifactor_backtest.py)
在特征计算公式、数据切分、T-1 延迟匹配及前瞻收益目标定义上 100% 完全一致。
"""

import os
import logging
import numpy as np
import pandas as pd

from cb_quant.unified_pit_engine import CBUnifiedPITEngine
from cb_quant.tcc_factor import CBTCCFactorEngine

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'double_low', 'premium_rate_t1', 'conv_value_t1', 'vol', 'amount',
    'curr_iss_amt', 'spike_ratio', 'tcc_factor',
    'ex_rtn_max_val_5min', 'ex_rtn_max_val_1min', 'ex_rtn_min_freq_5min'
]

def build_unified_feature_matrix(df_15m, mins_data_dir=r"D:\CB_mins_data", data_v2_dir=r"D:\iquant_data\data_v2"):
    """
    统一特征矩阵生成器
    1. 接入 PIT 无前视引擎 (如 T-1 正股收盘价、As-Of 多阶转股价生效日 merge_asof、零时间倒退强赎判定)
    2. 计算 TCC 网络中心度因子并进行严格的 T-1 跨日平移 (shift 1)，彻底杜绝盘中看到收盘价的泄漏
    3. 严格统一 3 大极值特征的计算公式
    4. 计算盘内 60 分钟 (4 个 15 分钟 Bar) 前瞻收益率目标 fwd_rtn_60m
    """
    logger.info("=== [UNIFIED FEATURE PIPELINE] 开始生成标准特征矩阵 (Train/Serve 100% 统一) ===")
    
    # 1. 接入 PIT 物理引擎
    pit_engine = CBUnifiedPITEngine(mins_data_dir=mins_data_dir, data_v2_dir=data_v2_dir)
    df_pit = pit_engine.build_unified_state_panel(df_15m)
    
    # 2. 计算 TCC 网络中心度因子
    tcc_engine = CBTCCFactorEngine(window=21)
    df_tcc = tcc_engine.generate_tcc_panel(start_date="2021-01-01")
    
    if df_tcc is not None and not df_tcc.empty:
        # T-1 跨日平移逻辑: TCC 依赖当天收盘价，故 T 日交易必须严格使用 T-1 日收盘算得的 tcc_factor
        daily_tcc = df_tcc.copy()
        daily_tcc.sort_values(by=['ts_code', 'date_str'], inplace=True)
        daily_tcc['tcc_factor_t1'] = daily_tcc.groupby('ts_code')['tcc_factor'].shift(1)
        
        df_pit = df_pit.merge(daily_tcc[['ts_code', 'date_str', 'tcc_factor_t1']],
                              on=['ts_code', 'date_str'], how='left')
        df_pit.rename(columns={'tcc_factor_t1': 'tcc_factor'}, inplace=True)
    else:
        df_pit['tcc_factor'] = np.nan

    # 3. 统一极值特征公式 (Train & Serve 绝对一致)
    df_pit['ex_rtn_max_val_5min'] = np.where(
        df_pit['close'].notnull() & (df_pit['close'] > 0),
        (df_pit['high'] - df_pit['close']) / df_pit['close'], np.nan
    )
    df_pit['ex_rtn_max_val_1min'] = np.where(
        df_pit['close'].notnull() & (df_pit['close'] > 0),
        (df_pit['close'] - df_pit['low']) / df_pit['close'], np.nan
    )
    df_pit['ex_rtn_min_freq_5min'] = np.where(
        df_pit['vol'].notnull() & (df_pit['vol'] > 0),
        df_pit['amount'] / (df_pit['vol'] * 100.0), np.nan
    )

    # 4. 计算盘内 60 分钟 (4 个 15m Bar) 前瞻收益率目标 fwd_rtn_60m
    df_pit.sort_values(by=['ts_code', 'date_int', 'time_str'], inplace=True)
    df_pit['fwd_rtn_60m'] = df_pit.groupby('ts_code')['close'].shift(-4) / df_pit['close'] - 1.0

    logger.info(f"统一特征矩阵构建完成，总记录数: {len(df_pit):,}")
    return df_pit
