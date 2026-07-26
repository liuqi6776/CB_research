# -*- coding: utf-8 -*-

"""
快速高效：时间网络相对中心度 (TCC) 因子在 A股可转债上的实证评估 (Fast TCC Factor Evaluation)
算法完全复现《股票网络与网络中心度因子研究》(曹春晓) 及文章代码：
1. 截面收益率 Z-Score: Z_{i,t} = (r_{i,t} - mean_t(r)) / std_t(r)
2. 偏离度平方: D_{i,t} = (Z_{i,t})^2
3. 21 日 Rolling Mean 倒数: TCC_{i,t} = 1 / mean_{21}(D_{i,t})
"""

import os
import sys
import numpy as np
import pandas as pd

def main():
    print("\n" + "="*80)
    print("   【时间网络相对中心度 (TCC / Time Network Relative Centrality) 实证报告】")
    print("="*80)
    
    # 加载日频数据或从 15m 面板提取日频收盘价
    from cb_quant.data_loader import CBDataLoader
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2024-12-01", end_date="2026-07-25", max_bonds=250)
    df_panel['date_str'] = pd.to_datetime(df_panel['trade_time']).dt.strftime('%Y%m%d')
    
    # 提取每日收盘价透视表 (index: date_str, columns: ts_code)
    daily_close = df_panel.groupby(['date_str', 'ts_code'])['close'].last().unstack()
    
    # 1. 计算日收益率
    rtn = daily_close.pct_change()
    
    # 2. 截面 Z-score 标准化偏离度
    mean_t = rtn.mean(axis=1)
    std_t = rtn.std(axis=1)
    z_score = rtn.sub(mean_t, axis=0).div(std_t, axis=0)
    
    # 3. 偏离度平方与 21 日 Rolling Inverse
    d_sq = np.square(z_score)
    roll_mean = d_sq.rolling(window=21, min_periods=5).mean()
    tcc = 1.0 / roll_mean
    
    # 未来 1 日收益率 (T+1 收益率作为标签，确保 T-1 因子对齐)
    fut_rtn = rtn.shift(-1)
    
    # 算 IC (TCC 与 fut_rtn 截面 Rank IC)
    ics = tcc.corrwith(fut_rtn, axis=1, method='spearman').dropna()
    mean_ic = ics.mean()
    std_ic = ics.std()
    ic_sharpe = mean_ic / (std_ic + 1e-8) * np.sqrt(252.0)
    positive_ic_ratio = (ics > 0).mean() * 100.0
    
    # 算 Q1~Q5 五分组收益
    tcc_long = tcc.stack().reset_index()
    tcc_long.columns = ['date_str', 'ts_code', 'tcc']
    fut_long = fut_rtn.stack().reset_index()
    fut_long.columns = ['date_str', 'ts_code', 'fut_rtn']
    
    df_merged = tcc_long.merge(fut_long, on=['date_str', 'ts_code']).dropna()
    df_merged['group'] = df_merged.groupby('date_str')['tcc'].transform(
        lambda x: pd.qcut(x.rank(method='first'), q=5, labels=['Q1(最低)', 'Q2', 'Q3', 'Q4', 'Q5(最高)']) if len(x)>=10 else np.nan
    )
    group_perf = df_merged.groupby('group')['fut_rtn'].mean() * 10000.0 # 单位: bp
    
    print("【1. IC 与 Rank IC 统计指标 (2025.01 ~ 2026.07)】")
    print("  - 平均 Rank IC:                 {:+.4f}".format(mean_ic))
    print("  - IC 标准差 (IC Std):            {:.4f}".format(std_ic))
    print("  - 年化 IC 夏普比率 (IC Sharpe):  {:+.2f}".format(ic_sharpe))
    print("  - IC 正胜率 (Positive IC %):     {:.1f}%".format(positive_ic_ratio))
    print("-" * 80)
    print("【2. Q1~Q5 五分组日均收益率 (单位: bp)】")
    for g, val in group_perf.items():
        print("  - {:12s}: {:+6.2f} bp".format(str(g), val))
    print("-" * 80)
    print("【3. 关键特性与应用结论】")
    if mean_ic > 0.02:
        print("  - 特性: TCC 因子在 A 股可转债上具备【正向选择效用】（稳健追踪型转债超额更显著）。")
    elif mean_ic < -0.02:
        print("  - 特性: TCC 因子在 A 股可转债上具备【反向选择效用】（高波动偏离型转债超额更显著）。")
    else:
        print("  - 特性: TCC 因子单因子 Rank IC 近似为 0，单靠 TCC 独立选债无法产生可持续超额。")
    print("  - 建议: 作为【高波动噪声过滤器】，在双低选债前剔除 TCC 最低的 20%-30% 极端偏离离群债。")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
