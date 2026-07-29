# -*- coding: utf-8 -*-

"""
绘制月频低换手可转债策略与 511380.SH 可转债 ETF 真实样本外 (OOS 2024.01 ~ 2026.06) 净值走势图
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from run_monthly_cb_etf_enhancement import run_monthly_exploration

REPO_ROOT = Path(__file__).resolve().parent

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def calc_max_drawdown(nav_series):
    """计算净值序列的最大回撤"""
    cummax = nav_series.cummax()
    dd = (nav_series - cummax) / cummax
    return dd.min()

def main():
    print("正在计算月频低换手策略与 511380.SH ETF 净值序列并绘制对比图...")
    
    res_monthly_dl, res_monthly_top5 = run_monthly_exploration()

    nav_m_dl = res_monthly_dl['nav_series'] / 1000000.0
    nav_m_top5 = res_monthly_top5['nav_series'] / 1000000.0

    u_dates_str = res_monthly_dl['u_dates']
    plot_dates = pd.to_datetime(u_dates_str, format='%Y%m%d')

    repo_artifacts_dir = REPO_ROOT / "artifacts"
    repo_artifacts_dir.mkdir(parents=True, exist_ok=True)
    etf_csv_path = repo_artifacts_dir / "cb_etf_511380_daily.csv"
    
    if etf_csv_path.exists():
        df_etf = pd.read_csv(etf_csv_path)
        df_etf['trade_date'] = pd.to_datetime(df_etf['日期'].astype(str))
        df_etf = df_etf.sort_values('trade_date').reset_index(drop=True)
        df_etf_sub = df_etf[df_etf['trade_date'].isin(plot_dates)].copy()
        nav_etf = df_etf_sub['收盘'] / df_etf_sub['收盘'].iloc[0]
        etf_total_ret = nav_etf.iloc[-1] - 1.0
        etf_max_dd = calc_max_drawdown(nav_etf)
    else:
        nav_etf = None

    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    
    if nav_etf is not None:
        ax.plot(plot_dates, nav_etf, label=f'511380.SH 可转债 ETF 真实基准 (OOS 累计 {etf_total_ret*100:+.2f}%, 回撤 {etf_max_dd*100:.2f}%)', color='#000000', linewidth=2.4, linestyle='--')

    ax.plot(plot_dates, nav_m_dl, label=f'月频纯双低 (Top 10) (OOS 累计 {res_monthly_dl["total_ret"]*100:+.2f}%, 夏普 {res_monthly_dl["sharpe"]:.2f}, 回撤 {res_monthly_dl["max_dd"]*100:.2f}%)', color='#059669', linewidth=2.0)
    ax.plot(plot_dates, nav_m_top5, label=f'月频精选双低 (Top 5) (OOS 累计 {res_monthly_top5["total_ret"]*100:+.2f}%, 夏普 {res_monthly_top5["sharpe"]:.2f}, 回撤 {res_monthly_top5["max_dd"]*100:.2f}%)', color='#1D4ED8', linewidth=1.8, linestyle='-.')

    ax.set_title("月频低换手可转债策略与 511380 可转债 ETF 真实样本外 (OOS 2024.01 ~ 2026.06) 净值对比图", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("真实交易日期 (Trade Date)", fontsize=11, labelpad=10)
    ax.set_ylabel("归一化净值 (Empirical NAV, 初始 = 1.0000)", fontsize=11, labelpad=10)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

    plt.tight_layout()
    
    out_path_repo = repo_artifacts_dir / "monthly_strategy_vs_etf_nav.png"
    plt.savefig(out_path_repo, dpi=300)
    print(f"Monthly Strategy NAV Chart saved to:\n  - Repo Artifact: {out_path_repo}")

if __name__ == '__main__':
    main()
