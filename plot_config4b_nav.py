# -*- coding: utf-8 -*-

"""
绘制 2024.01 ~ 2026.06 (全周期) 全量多因子 GBDT 策略、诚实纯双低与 511380.SH 可转债 ETF 真实基准 (OOS) 净值走势图
使用跨平台 pathlib 相对路径、严格消费物理行情交易日历 (u_dates)，结合博时中证可转债 ETF 真实行情。
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from run_master_multifactor_backtest import run_empirical_backtest

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
    print("正在从物理行情引擎提取真实样本外 (OOS) 动态净值序列与 511380 CB ETF 基准并绘制图表...")
    
    res_base, res_cfg1, res_cfg2, res_cfg3, res_cfg4a, res_cfg4b = run_empirical_backtest()

    nav_base = res_base['nav_series'] / 1000000.0
    nav_cfg1 = res_cfg1['nav_series'] / 1000000.0
    nav_cfg2 = res_cfg2['nav_series'] / 1000000.0
    nav_cfg3 = res_cfg3['nav_series'] / 1000000.0
    nav_cfg4a = res_cfg4a['nav_series'] / 1000000.0
    nav_cfg4b = res_cfg4b['nav_series'] / 1000000.0
    nav_portfolio = 0.80 * 1.0 + 0.20 * nav_cfg4b

    max_dd_portfolio = calc_max_drawdown(nav_portfolio)
    total_ret_portfolio = nav_portfolio.iloc[-1] - 1.0

    u_dates_str = res_base['u_dates']
    plot_dates = pd.to_datetime(u_dates_str, format='%Y%m%d')

    # 读取 511380.SH (博时中证可转债 ETF) 真实基准行情
    repo_artifacts_dir = REPO_ROOT / "artifacts"
    repo_artifacts_dir.mkdir(parents=True, exist_ok=True)
    etf_csv_path = repo_artifacts_dir / "cb_etf_511380_daily.csv"
    
    if etf_csv_path.exists():
        df_etf = pd.read_csv(etf_csv_path)
        df_etf['trade_date'] = pd.to_datetime(df_etf['日期'].astype(str))
        df_etf = df_etf.sort_values('trade_date').reset_index(drop=True)
        # 对齐 598 个交易日
        df_etf_sub = df_etf[df_etf['trade_date'].isin(plot_dates)].copy()
        nav_etf = df_etf_sub['收盘'] / df_etf_sub['收盘'].iloc[0]
        etf_total_ret = nav_etf.iloc[-1] - 1.0
        etf_max_dd = calc_max_drawdown(nav_etf)
    else:
        nav_etf = None

    df_oos_nav = pd.DataFrame({
        'trade_date': plot_dates,
        'Config_0_Baseline': nav_base.values,
        'Config_1_TCC': nav_cfg1.values,
        'Config_2_GBDT_OOS': nav_cfg2.values,
        'Config_4b_3Tier_Timing': nav_cfg4b.values,
        'Config_5_Portfolio_80_20': nav_portfolio.values,
        '511380_CB_ETF_Benchmark': nav_etf.values if nav_etf is not None else np.nan
    })
    
    csv_out_repo = repo_artifacts_dir / "nav_results_oos.csv"
    df_oos_nav.to_csv(csv_out_repo, index=False, encoding='utf-8-sig')
    print(f"已全量导出样本外 100% 真实净值序列产物 (共 {len(df_oos_nav)} 个真实交易日, 终点: {u_dates_str[-1]}): {csv_out_repo}")

    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    
    if nav_etf is not None:
        ax.plot(plot_dates, nav_etf, label=f'511380.SH 可转债 ETF 真实基准 (OOS 累计 {etf_total_ret*100:+.2f}%, 回撤 {etf_max_dd*100:.2f}%)', color='#000000', linewidth=2.4, linestyle='--')

    ax.plot(plot_dates, nav_base, label=f'Config 0: 纯双低诚实基准 (OOS 累计 {res_base["total_ret"]*100:+.2f}%, 夏普 {res_base["sharpe"]:.2f}, 回撤 {res_base["max_dd"]*100:.2f}%)', color='#DC2626', linewidth=2.0)
    ax.plot(plot_dates, nav_cfg2, label=f'Config 2: GBDT 14大因子 OOS 预测 (OOS 累计 {res_cfg2["total_ret"]*100:+.2f}%, 夏普 {res_cfg2["sharpe"]:.2f}, 回撤 {res_cfg2["max_dd"]*100:.2f}%)', color='#1D4ED8', linewidth=1.8)
    ax.plot(plot_dates, nav_cfg1, label=f'Config 1: 纯双低 + TCC 因子过滤 (OOS 累计 {res_cfg1["total_ret"]*100:+.2f}%, 夏普 {res_cfg1["sharpe"]:.2f}, 回撤 {res_cfg1["max_dd"]*100:.2f}%)', color='#059669', linewidth=1.8, linestyle=':')
    ax.plot(plot_dates, nav_cfg4b, label=f'Config 4b: GBDT + 三档动态控仓 (OOS 累计 {res_cfg4b["total_ret"]*100:+.2f}%, 夏普 {res_cfg4b["sharpe"]:.2f}, 回撤 {res_cfg4b["max_dd"]*100:.2f}%)', color='#7C3AED', linewidth=1.8, linestyle='-.')
    ax.plot(plot_dates, nav_portfolio, label=f'Config 5: 80/20 组合部署框架 (OOS 累计 {total_ret_portfolio*100:+.2f}%, 回撤 {max_dd_portfolio*100:.2f}%)', color='#D97706', linewidth=1.8)

    ax.set_title("全量多因子 GBDT 策略与 511380 可转债 ETF 真实样本外 (OOS 2024.01 ~ 2026.06) 净值对比图", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("真实交易日期 (Trade Date)", fontsize=11, labelpad=10)
    ax.set_ylabel("归一化净值 (Empirical NAV, 初始 = 1.0000)", fontsize=11, labelpad=10)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

    plt.tight_layout()
    
    out_path_repo = repo_artifacts_dir / "config4b_vs_etf_nav.png"
    plt.savefig(out_path_repo, dpi=300)
    print(f"Empirical OOS NAV Chart saved to:\n  - Repo Artifact: {out_path_repo}")

if __name__ == '__main__':
    main()
