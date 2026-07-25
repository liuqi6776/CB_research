# -*- coding: utf-8 -*-

"""
可转债分钟线数据加载与面板构建模块
Convertible Bond Minute-Level Data Ingestion & Panel Builder
"""

import os
import glob
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class CBDataLoader:
    def __init__(self, data_dir=r"D:\CB_mins_data"):
        self.data_dir = data_dir
        self.parquet_dir = os.path.join(data_dir, "parquet")
        self.basic_info_path = os.path.join(data_dir, "cb_basic_info.csv")
        
    def load_basic_info(self):
        """加载全量可转债基础元数据"""
        if os.path.exists(self.basic_info_path):
            df = pd.read_csv(self.basic_info_path)
            return df
        return pd.DataFrame()

    def load_minute_panel(self, start_date="2024-01-01", end_date="2026-07-25", max_bonds=None):
        """
        加载指定时间段内的所有可转债 5 分钟 K 线数据，并构建对齐面板
        """
        parquet_files = glob.glob(os.path.join(self.parquet_dir, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"未在 {self.parquet_dir} 找到任何 parquet 文件。")
            
        logger.info(f"搜索到 {len(parquet_files)} 只可转债行情文件，开始清洗与拼接...")
        
        all_dfs = []
        if max_bonds and max_bonds > 0:
            parquet_files = parquet_files[:max_bonds]

        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                if df.empty or 'trade_time' not in df.columns:
                    continue
                
                df['trade_time'] = pd.to_datetime(df['trade_time'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
                df = df.dropna(subset=['trade_time'])
                # 过滤时间段
                df = df[(df['trade_time'] >= start_date) & (df['trade_time'] <= end_date)].copy()
                if df.empty:
                    continue
                
                # 转换数值类型
                for col in ['open', 'high', 'low', 'close', 'vol', 'amount']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                all_dfs.append(df[['ts_code', 'trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount']])
            except Exception as e:
                logger.warning(f"读取文件 {f} 失败: {e}")

        if not all_dfs:
            raise ValueError(f"指定时间段 ({start_date} ~ {end_date}) 内无可用行情数据。")

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df = full_df.sort_values(by=['trade_time', 'ts_code']).reset_index(drop=True)
        
        logger.info(f"成功加载行情总记录数: {len(full_df):,} 行，覆盖可转债: {full_df['ts_code'].nunique()} 只。")
        return full_df
