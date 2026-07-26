# -*- coding: utf-8 -*-

"""
日频筹码分布因子生成器 (Daily Stock Chip Factor Generator)
功能：计算正股筹码集中度、获利盘比例与 20 日成本分位，输出 daily_chip.parquet
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class DailyChipFactorEngine:
    """
    正股日频筹码与成本分布因子引擎
    """
    def __init__(self, data_v2_dir=r"D:\iquant_data\data_v2", mins_data_dir=r"D:\CB_mins_data"):
        self.data_v2_dir = data_v2_dir
        self.mins_data_dir = mins_data_dir

    def generate_daily_chip_factors(self, output_path="daily_chip.parquet"):
        """
        计算正股日频筹码分布因子
        - profit_ratio: 股价相对 20 日 VWAP 获利比例
        - cost_concentration_90: 20 日价格波动带相对集中度
        - cost_position_20d: 股价在过去 20 日成交密集区中的相对分位
        """
        logging.info("=== 开始生成正股日频筹码与成本分布因子 ===")
        day_files = sorted(glob.glob(os.path.join(self.data_v2_dir, "data_day1", "*.parquet")))
        valid_day_files = [f for f in day_files if os.path.basename(f).replace('.parquet','') >= '20241201']
        
        if not valid_day_files:
            logging.error("未找到有效的日线 parquet 数据！")
            return pd.DataFrame()

        day_dfs = []
        for f in valid_day_files:
            try:
                df_sub = pd.read_parquet(f, columns=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount'])
                day_dfs.append(df_sub)
            except Exception:
                pass

        if not day_dfs:
            return pd.DataFrame()

        stk_df = pd.concat(day_dfs, ignore_index=True)
        stk_df.rename(columns={'ts_code': 'stk_code'}, inplace=True)
        stk_df['trade_date_str'] = stk_df['trade_date'].astype(str)
        stk_df = stk_df.sort_values(by=['stk_code', 'trade_date_str']).reset_index(drop=True)

        stk_df['vwap'] = stk_df['amount'] / (stk_df['vol'] + 1e-8)
        stk_df['vwap_20d'] = stk_df.groupby('stk_code')['amount'].rolling(20, min_periods=5).sum().reset_index(level=0, drop=True) / \
                             (stk_df.groupby('stk_code')['vol'].rolling(20, min_periods=5).sum().reset_index(level=0, drop=True) + 1e-8)

        stk_df['high_20d'] = stk_df.groupby('stk_code')['high'].rolling(20, min_periods=5).max().reset_index(level=0, drop=True)
        stk_df['low_20d'] = stk_df.groupby('stk_code')['low'].rolling(20, min_periods=5).min().reset_index(level=0, drop=True)

        stk_df['chip_profit_ratio'] = (stk_df['close'] - stk_df['vwap_20d']) / (stk_df['vwap_20d'] + 1e-8)
        stk_df['chip_concentration_90'] = (stk_df['high_20d'] - stk_df['low_20d']) / (stk_df['vwap_20d'] + 1e-8)
        stk_df['chip_position_20d'] = (stk_df['close'] - stk_df['low_20d']) / (stk_df['high_20d'] - stk_df['low_20d'] + 1e-8)

        chip_df = stk_df[['stk_code', 'trade_date_str', 'chip_profit_ratio', 'chip_concentration_90', 'chip_position_20d']].copy()
        chip_df.dropna(subset=['chip_profit_ratio'], inplace=True)

        if output_path.endswith('.parquet'):
            chip_df.to_parquet(output_path, index=False)
        else:
            chip_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        logging.info(f"日频筹码因子已成功输出至: {output_path} (共 {len(chip_df):,} 条记录)")
        return chip_df

if __name__ == '__main__':
    engine = DailyChipFactorEngine()
    engine.generate_daily_chip_factors()
