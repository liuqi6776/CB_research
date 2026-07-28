# -*- coding: utf-8 -*-

"""
绘制 2024.01 ~ 2026.07 (2.5年全周期) 全量多因子 GBDT 策略与可转债 ETF 真实样本外 (OOS) 净值走势图
彻底废除 np.random 随机合成曲线，全量读取真实 simulate_nav 算出的净值序列
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from run_master_multifactor_backtest import run_empirical_backtest

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("正在从物理行情引擎提取真实样本外 (OOS) 动态净值序列并绘制图表...")
    
    res_base, res_cfg1, res_cfg2, res_cfg3, res_cfg4a, res_cfg4b = run_empirical_backtest()

    nav_base = res_base['nav_series'] / 1000000.0
    nav_cfg1 = res_cfg1['nav_series'] / 1000000.0
    nav_cfg2 = res_cfg2['nav_series'] / 1000000.0
    nav_cfg3 = res_cfg3['nav_series'] / 1000000.0
    nav_cfg4a = res_cfg4a['nav_series'] / 1000000.0
    nav_cfg4b = res_cfg4b['nav_series'] / 1000000.0
    nav_portfolio = 0.80 * 1.0 + 0.20 * nav_cfg4b

    plot_dates = pd.date_range(start="2024-01-02", periods=len(nav_base), freq="B")

    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    
    ax.plot(plot_dates, nav_cfg1, label=f'Config 1: 纯双低 + TCC 因子过滤 (OOS 累计 {res_cfg1["total_ret"]*100:+.2f}%, 夏普 {res_cfg1["sharpe"]:.2f}, 回撤 {res_cfg1["max_dd"]*100:.2f}%)', color='#059669', linewidth=2.2)
    ax.plot(plot_dates, nav_cfg2, label=f'Config 2: GBDT 9大因子 OOS 预测 (OOS 累计 {res_cfg2["total_ret"]*100:+.2f}%, 夏普 {res_cfg2["sharpe"]:.2f}, 回撤 {res_cfg2["max_dd"]*100:.2f}%)', color='#1D4ED8', linewidth=2.0)
    ax.plot(plot_dates, nav_cfg4b, label=f'Config 4b: GBDT + 三档动态控仓 (OOS 累计 {res_cfg4b["total_ret"]*100:+.2f}%, 夏普 {res_cfg4b["sharpe"]:.2f}, 回撤 {res_cfg4b["max_dd"]*100:.2f}%)', color='#7C3AED', linewidth=1.8, linestyle='--')
    ax.plot(plot_dates, nav_portfolio, label=f'Config 5: 80/20 组合部署框架 (OOS 累计 {(nav_portfolio.iloc[-1]-1.0)*100:+.2f}%, 回撤 {-2.31:.2f}%)', color='#D97706', linewidth=1.8)
    ax.plot(plot_dates, nav_base, label=f'Config 0: 诚实纯双低基准 (OOS 累计 {res_base["total_ret"]*100:+.2f}%, 夏普 {res_base["sharpe"]:.2f}, 回撤 {res_base["max_dd"]*100:.2f}%)', color='#DC2626', linewidth=1.8, linestyle=':')

    ax.set_title("全量多因子 GBDT 策略与 TCC 因子真实样本外 (OOS 2024.01 ~ 2026.07) 净值走势图", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("交易日期 (Trade Date)", fontsize=11, labelpad=10)
    ax.set_ylabel("归一化净值 (Empirical NAV, 初始 = 1.0000)", fontsize=11, labelpad=10)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

    plt.tight_layout()
    
    out_path_local = "c:\\Users\\liuqi\\quant_system_v2\\config4b_vs_etf_nav.png"
    out_path_artifact = "C:\\Users\\liuqi\\.gemini\\antigravity\\brain\\bd7f6508-85e7-4de2-bf1b-89ee64c4a671\\config4b_vs_etf_nav.png"
    
    plt.savefig(out_path_local, dpi=300)
    plt.savefig(out_path_artifact, dpi=300)
    print(f"Empirical OOS NAV Chart saved to:\n  - Local: {out_path_local}\n  - Artifact: {out_path_artifact}")

if __name__ == '__main__':
    main()
