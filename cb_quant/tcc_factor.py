# -*- coding: utf-8 -*-

"""
时间网络相对中心度因子引擎 (Time Network Relative Centrality / TCC Factor Engine)
参考论文: 曹春晓《股票网络与网络中心度因子研究》
算法逻辑:
1. 截面收益率 Z-Score: Z_{i,t} = (r_{i,t} - mean_t(r)) / std_t(r)
2. 相对偏离度平方: D_{i,t} = (Z_{i,t})^2
3. 21 日滚动均值倒数: TCC_{i,t} = 1 / mean_{21}(D_{i,t})
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class CBTCCFactorEngine:
    """
    时间网络相对中心度 (TCC) 因子计算器
    """
    def __init__(self, window=21):
        self.window = window

    def compute_tcc_factor(self, df_daily_close):
        """
        输入: 透视矩阵 df_daily_close (index: datetime/date_str, columns: ts_code)
        输出: TCC 因子透视矩阵 (与 df_daily_close 结构完全相同)
        """
        # 1. 计算日收益率
        rtn = df_daily_close.pct_change()
        
        # 2. 截面 Z-Score 标准化
        mean_t = rtn.mean(axis=1)
        std_t = rtn.std(axis=1)
        
        z_score = rtn.sub(mean_t, axis=0).div(std_t, axis=0)
        
        # 3. 偏离度平方
        d_sq = np.square(z_score)
        
        # 4. 21 日 Rolling Mean 的倒数
        roll_mean = d_sq.rolling(window=self.window, min_periods=5).mean()
        tcc = 1.0 / roll_mean
        
        return tcc

    def generate_tcc_panel(self, start_date="2024-12-01", end_date="2026-07-25"):
        logging.info("=== 开始构建时间网络相对中心度 (TCC) 因子面板 ===")
        loader = CBDataLoader()
        df_panel = loader.load_minute_panel(start_date=start_date, end_date=end_date, max_bonds=250)
        
        # 提取每日收盘价透视表
        df_panel['date_str'] = pd.to_datetime(df_panel['trade_time']).dt.strftime('%Y%m%d')
        daily_close = df_panel.groupby(['date_str', 'ts_code'])['close'].last().unstack()
        
        tcc_df = self.compute_tcc_factor(daily_close)
        
        # 转为长表格式
        tcc_long = tcc_df.stack().reset_index()
        tcc_long.columns = ['date_str', 'ts_code', 'tcc_factor']
        tcc_long.dropna(subset=['tcc_factor'], inplace=True)
        
        logging.info(f"TCC 因子构建完成，共生成 {len(tcc_long):,} 条观测记录。")
        return tcc_long

if __name__ == '__main__':
    engine = CBTCCFactorEngine(window=21)
    df_tcc = engine.generate_tcc_panel()
    print(df_tcc.head(10))
