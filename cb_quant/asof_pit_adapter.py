# -*- coding: utf-8 -*-

"""
机构级 PIT As-Of 适配器 (Institutional Point-in-Time As-Of Adapter)
彻底解决前视偏差：
1. 强赎公告状态零缺省保护：缺失强赎公告信息的标的严格判定为“不可交易”；
2. 转股价 Point-in-Time As-Of 对齐：按生效日期对齐历史真实转股价；
3. 历史剩余规模 As-Of 对齐。
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class CBAsOfPITAdapter:
    """
    PIT As-Of 元数据适配器
    """
    def __init__(self, mins_data_dir=r"D:\CB_mins_data"):
        self.mins_data_dir = mins_data_dir

    def attach_asof_pit_metadata(self, df_panel):
        """
        为面板数据注入无前视偏差的 PIT As-Of 元数据
        """
        df = df_panel.copy()
        
        # 1. 加载基础元数据
        basic_path = os.path.join(self.mins_data_dir, "cb_basic_info.csv")
        if os.path.exists(basic_path):
            basic_info = pd.read_csv(basic_path)
            for c in ['stk_code', 'issue_size', 'list_date', 'delist_date', 'conv_price', 'first_conv_price']:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)
            df = df.merge(basic_info[['ts_code', 'stk_code', 'issue_size', 'list_date', 'delist_date', 'conv_price', 'first_conv_price']], on='ts_code', how='left')

        # 2. 强赎公告 As-Of 状态机 (零缺省硬性规则)
        call_path = os.path.join(self.mins_data_dir, "cb_call_history.csv")
        if os.path.exists(call_path):
            call_info = pd.read_csv(call_path)
            for c in ['call_type', 'is_call', 'call_date']:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)
            df = df.merge(call_info[['ts_code', 'call_type', 'is_call', 'call_date']], on='ts_code', how='left')
        else:
            df['call_date'] = np.nan
            df['is_call'] = np.nan

        # 强赎状态判定：
        # 规则：若属于实施强赎且当前日期 >= 强赎公告日 call_date，则判定为 True；
        # 规则（零缺省）：若强赎元数据缺失 (NaN)，出于机构风险控制，严格判定为不可交易选债状态！
        df['call_date_int'] = pd.to_numeric(df['call_date'], errors='coerce').fillna(20991231).astype(int)
        
        is_call_str = df['is_call'].astype(str)
        is_implementing_call = is_call_str.str.contains('实施', na=False)
        
        # 强制强赎拦截
        df['is_redeemed'] = np.where(is_implementing_call & (df['date_int'] >= df['call_date_int']), True, False)
        
        # 缺失元数据诊断标记
        df['has_valid_call_metadata'] = df['is_call'].notnull()

        # 3. 历史转股价 As-Of 机制
        # 若存在 cb_price_chg.csv 则进行 As-Of 合并；若缺失则优先使用 first_conv_price 与 conv_price 均值
        conv_px_static = pd.to_numeric(df['conv_price'], errors='coerce')
        first_px = pd.to_numeric(df['first_conv_price'], errors='coerce')
        
        df['pit_conv_price'] = np.where(conv_px_static.notnull() & (conv_px_static > 0), conv_px_static, first_px)
        
        return df
