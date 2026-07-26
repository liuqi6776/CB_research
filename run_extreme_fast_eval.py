# -*- coding: utf-8 -*-

"""
超高速：收益率极大值幅度 (Extreme Return Magnitude / ex_rtn_max_val) 实证评估
参考研报: 兴业证券 郑兆磊《高频研究系列三：收益率分布中的Alpha（2）》
计算 4 种衍生变体:
1. ex_rtn_max_val_5min  (5分钟极大值幅度)
2. ex_rtn_min_freq_5min (5分钟极小值频率)
3. ex_rtn_max_val_1min  (1分钟极大值幅度)
4. ex_rtn_min_freq_1min (1分钟极小值频率)
"""

import os
import sys
import numpy as np
import pandas as pd
from cb_quant.data_loader import CBDataLoader
from cb_quant.extreme_return_factor import CBExtremeReturnFactorEngine

def main():
    print("\n" + "="*85)
    print("   【收益率极大值幅度 (Extreme Return Magnitude / ex_rtn_max_val) 实证报告】")
    print("="*85)
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    # 1. 矢量化计算因子
    erm_engine = CBExtremeReturnFactorEngine()
    df_erm = erm_engine.generate_extreme_return_panel(df_panel)
    
    # 未来 1 日收益率标签 (fut_rtn)
    df_panel['date_str'] = pd.to_datetime(df_panel['trade_time']).dt.strftime('%Y%m%d')
    daily_close = df_panel.groupby(['date_str', 'ts_code'])['close'].last().unstack()
    daily_rtn = daily_close.pct_change()
    fut_rtn = daily_rtn.shift(-1)
    
    fut_long = fut_rtn.stack().reset_index()
    fut_long.columns = ['date_str', 'ts_code', 'fut_rtn']
    
    # T-1 对齐: 下一交易日使用的因子值来自于 T-1
    unique_dates = sorted(df_erm['date_str'].unique())
    date_map = {unique_dates[i]: unique_dates[i+1] for i in range(len(unique_dates)-1)}
    df_erm['t1_trade_date'] = df_erm['date_str'].map(date_map)
    
    df_eval = df_erm.merge(fut_long, left_on=['t1_trade_date', 'ts_code'], right_on=['date_str', 'ts_code'], suffixes=('', '_y')).dropna(subset=['fut_rtn'])
    
    # 2. 4 种因子的 Rank IC 指标诊断 (向量化)
    f_cols = ['ex_rtn_max_val_5min', 'ex_rtn_min_freq_5min', 'ex_rtn_max_val_1min', 'ex_rtn_min_freq_1min']
    
    print("【1. 收益率极大值/极小值 4 种衍生因子的截面 Rank IC 表现 (2025.01 ~ 2026.07)】")
    print("因子名称                   | 平均 Rank IC | IC 标准差 | 年化 IC 夏普 | IC 正胜率 | 作用方向评估")
    print("-" * 85)
    
    best_factor = None
    best_sharpe = -999.0
    
    for f in f_cols:
        # 透视计算 corrwith
        p_factor = df_eval.pivot(index='t1_trade_date', columns='ts_code', values=f)
        p_label = df_eval.pivot(index='t1_trade_date', columns='ts_code', values='fut_rtn')
        
        ics = p_factor.corrwith(p_label, axis=1, method='spearman').dropna()
        m_ic = ics.mean()
        s_ic = ics.std()
        sh_ic = m_ic / (s_ic + 1e-8) * np.sqrt(252.0)
        win_ic = (ics > 0).mean() * 100.0
        
        eval_str = "强多头 (超额显著)" if sh_ic > 1.5 else ("强空头 (反向有效)" if sh_ic < -1.5 else "中性/弱相关")
        
        print("{:26s} | {:+10.4f}  | {:8.4f}  | {:+11.2f}  | {:8.1f}%  | {}".format(
            f, m_ic, s_ic, sh_ic, win_ic, eval_str
        ))
        
        if sh_ic > best_sharpe:
            best_sharpe = sh_ic
            best_factor = f
            
    print("-" * 85)
    
    # 3. 最佳因子的 Q1~Q5 五分组收益率
    df_eval['group'] = df_eval.groupby('t1_trade_date')[best_factor].transform(
        lambda x: pd.qcut(x.rank(method='first'), q=5, labels=['Q1(最低)', 'Q2', 'Q3', 'Q4', 'Q5(最高)']) if len(x)>=10 else np.nan
    )
    group_perf = df_eval.groupby('group')['fut_rtn'].mean() * 10000.0
    
    print(f"【2. 最佳因子 [{best_factor}] Q1~Q5 五分组日均收益率 (单位: bp)】")
    for g, val in group_perf.items():
        print("  - {:12s}: {:+6.2f} bp".format(str(g), val))
        
    print("-" * 85)
    print("【3. 策略评估与总结】")
    print(f"  - 最优衍生因子: [{best_factor}]，年化 IC 夏普达到 {best_sharpe:+.2f}；")
    print("  - 建议: 将该因子纳入有效因子候选库，后续统一用于多因子 GBDT / 线性组合建模。")
    print("="*85 + "\n")

if __name__ == '__main__':
    main()
