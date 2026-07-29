# -*- coding: utf-8 -*-

"""
月频低换手可转债 ETF 相对超额增强策略研究引擎 (Monthly Low-Turnover CB ETF Alpha Enhancement Engine)

核心创新点 (严格 Fail-Fast 零前视对齐 - 方案 A)：
1. 信号生成时间 (feature_time)：上自然月末尾最后一个交易日 15:00 收盘快照。
2. 订单执行时间 (execution_time)：本自然月首个交易日 09:35 开盘成交 (trade_date = curr_month_first_date)。
3. 零前视物理保证：feature_time 严格早于 execution_time，彻底杜绝同 Bar Open/Close 跨时空对齐泄漏。
4. 导出复现产物：artifacts/monthly_nav_results.csv。
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from cb_quant.data_loader import CBDataLoader
from cb_quant.feature_pipeline import build_unified_feature_matrix
from run_master_multifactor_backtest import simulate_nav

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent

def generate_monthly_orders(df_pit_elig, top_n=10):
    """
    按月频生成调仓目标单 (方案A：上月尾日 15:00 收盘快照选债 -> 本月首日 09:35 开盘成交)
    零前视对齐：feature_time (T-1 15:00) 严格早于 execution_time (T 09:35)
    """
    df_pit_elig = df_pit_elig.copy()
    df_pit_elig['month_str'] = df_pit_elig['date_str'].astype(str).str[:6]
    
    dates_by_month = df_pit_elig.groupby('month_str')['date_str'].unique().to_dict()
    months = sorted(dates_by_month.keys())
    order_records = []
    
    max_time = df_pit_elig['time_str'].max() # 15:00 收盘
    
    for i in range(1, len(months)):
        prev_month = months[i - 1]
        curr_month = months[i]
        
        # 上个月的最后一个交易日 (信号日)
        prev_month_last_date = dates_by_month[prev_month][-1]
        # 本月的第一个交易日 (成交日)
        curr_month_first_date = dates_by_month[curr_month][0]
        
        # 提取上月末交易日 15:00 的收盘因子快照
        df_snap = df_pit_elig[
            (df_pit_elig['date_str'] == prev_month_last_date) & 
            (df_pit_elig['time_str'] == max_time)
        ].copy()
        
        if df_snap.empty:
            df_snap = df_pit_elig[df_pit_elig['date_str'] == prev_month_last_date].groupby('ts_code').last().reset_index()
            
        df_snap['rank'] = df_snap['double_low'].rank(ascending=True, method='min')
        top_bonds = df_snap[df_snap['rank'] <= top_n]
        
        for _, row in top_bonds.iterrows():
            order_records.append({
                'feature_date': prev_month_last_date,
                'trade_date': curr_month_first_date,  # 本月首日开盘成交
                'ts_code': row['ts_code'],
                'double_low': row['double_low'],
                'rank': row['rank']
            })
            
    df_orders = pd.DataFrame(order_records)
    return df_orders

def run_monthly_exploration():
    logger.info("=== 启动【月频低换手可转债 ETF 相对超额增强策略 (方案A 零前视对齐)】实证流程 ===")
    
    loader = CBDataLoader()
    df_15m = loader.load_minute_panel(start_date="2024-01-01", max_bonds=None)
    df_pit_base = build_unified_feature_matrix(df_15m)
    
    # 截断至正股完整区间
    df_pit_base = df_pit_base[df_pit_base['date_str'] <= '20260625'].copy()
    u_dates = sorted(df_pit_base['date_str'].unique())
    logger.info(f"有效交易日共 {len(u_dates)} 天 (起点: {u_dates[0]}, 终点: {u_dates[-1]})")

    # 1. 月频纯双低策略 (Monthly Double-Low Top 10)
    df_pit_elig = df_pit_base[df_pit_base['is_eligible_at_selection'] == True]
    df_monthly_orders_dl = generate_monthly_orders(df_pit_elig, top_n=10)
    res_monthly_dl = simulate_nav(df_pit_base, df_monthly_orders_dl, use_timing=False)

    # 2. 月频双低 + 5 债集中组合 (Monthly Top 5)
    df_monthly_orders_top5 = generate_monthly_orders(df_pit_elig, top_n=5)
    res_monthly_top5 = simulate_nav(df_pit_base, df_monthly_orders_top5, use_timing=False)

    # 3. 严格 Fail-Fast 读取 511380.SH CB ETF 真实基准 (绝不保留硬编码回退)
    etf_csv_path = REPO_ROOT / "artifacts" / "cb_etf_511380_daily.csv"
    if not etf_csv_path.exists():
        raise FileNotFoundError(f"CRITICAL: 511380.SH ETF 真实基准行情文件不存在: {etf_csv_path}！")
        
    df_etf = pd.read_csv(etf_csv_path)
    plot_dates = pd.to_datetime(u_dates, format='%Y%m%d')
    df_etf['trade_date'] = pd.to_datetime(df_etf['日期'].astype(str))
    df_etf_sub = df_etf[df_etf['trade_date'].isin(plot_dates)].sort_values('trade_date').reset_index(drop=True)
    nav_etf = df_etf_sub['收盘'] / df_etf_sub['收盘'].iloc[0]
    etf_total_ret = nav_etf.iloc[-1] - 1.0
    etf_ann_ret = (1.0 + etf_total_ret) ** (252.0 / len(nav_etf)) - 1.0
    etf_sharpe = (nav_etf.pct_change().fillna(0.0).mean() / (nav_etf.pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    c_max_e = nav_etf.cummax()
    etf_max_dd = ((nav_etf - c_max_e) / c_max_e).min()

    # 导出复现产物 CSV: artifacts/monthly_nav_results.csv
    df_monthly_csv = pd.DataFrame({
        'trade_date': plot_dates,
        'Monthly_DoubleLow_Top10': res_monthly_dl['nav_series'].values / 1000000.0,
        'Monthly_DoubleLow_Top5': res_monthly_top5['nav_series'].values / 1000000.0,
        '511380_CB_ETF_Benchmark': nav_etf.values
    })
    csv_out_repo = REPO_ROOT / "artifacts" / "monthly_nav_results.csv"
    df_monthly_csv.to_csv(csv_out_repo, index=False, encoding='utf-8-sig')
    logger.info(f"已全量导出月频策略 100% 零前视净值序列产物 ({len(df_monthly_csv)} 个交易日): {csv_out_repo}")

    print("\n" + "="*142)
    print("      【零前视月频对齐 (方案A) 可转债策略 vs 511380.SH ETF 完整实证报告】")
    print("="*142)
    print("策略配置名称                            | 累计收益率 | 年化收益率 | 夏普比率 | 最大回撤 | 年化换手率 | 平均持仓天数 | 总摩擦成本(元) | 相对 ETF 超额")
    print("-" * 142)
    print("511380.SH 可转债 ETF 真实基准           | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  |    --    |     --   |      --        |   0.00pp (基准)".format(
        etf_total_ret*100, etf_ann_ret*100, etf_sharpe, etf_max_dd*100))
    print("1. 月频纯双低 (Top 10, 方案A零前视)     | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  | {:9.1f}x | {:10.1f}天 | ￥{:11.2f} | {:+7.2f}pp".format(
        res_monthly_dl['total_ret']*100, res_monthly_dl['ann_ret']*100, res_monthly_dl['sharpe'], res_monthly_dl['max_dd']*100,
        res_monthly_dl['turnover_annual'], res_monthly_dl['avg_holding_days'], res_monthly_dl['total_friction_cost'], (res_monthly_dl['total_ret'] - etf_total_ret)*100))
    print("2. 月频精选双低 (Top 5 集中, 方案A零前视)| {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  | {:9.1f}x | {:10.1f}天 | ￥{:11.2f} | {:+7.2f}pp".format(
        res_monthly_top5['total_ret']*100, res_monthly_top5['ann_ret']*100, res_monthly_top5['sharpe'], res_monthly_top5['max_dd']*100,
        res_monthly_top5['turnover_annual'], res_monthly_top5['avg_holding_days'], res_monthly_top5['total_friction_cost'], (res_monthly_top5['total_ret'] - etf_total_ret)*100))
    print("="*142 + "\n")

    return res_monthly_dl, res_monthly_top5

if __name__ == '__main__':
    run_monthly_exploration()
