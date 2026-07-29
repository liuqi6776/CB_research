# -*- coding: utf-8 -*-

"""
绘制机构不同手续费费率梯度 (0 bps -> 20 bps) 下可转债策略收益与夏普比率敏感度矩阵曲线图
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from run_institutional_friction_sensitivity import run_sensitivity_analysis

REPO_ROOT = Path(__file__).resolve().parent

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("正在计算机构费率敏感度矩阵数据并绘制对比曲线...")
    df_res, etf_total_ret, etf_sharpe = run_sensitivity_analysis()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    strategies = df_res['Strategy'].unique()
    colors = ['#059669', '#1D4ED8', '#D97706']
    markers = ['o', 's', '^']

    # 1. 累计收益率敏感度曲线
    for idx, strat in enumerate(strategies):
        sub = df_res[df_res['Strategy'] == strat].sort_values('Round_Trip_Fee_bps')
        ax1.plot(sub['Round_Trip_Fee_bps'], sub['Total_Return'] * 100, label=strat, color=colors[idx], marker=markers[idx], linewidth=2.0)
        
    ax1.axhline(y=etf_total_ret * 100, color='#000000', linestyle='--', linewidth=2.0, label=f'511380.SH ETF 基准 ({etf_total_ret*100:+.2f}%)')
    ax1.set_title("可转债策略累计收益率 vs 往返交易摩擦费率 (0 ~ 20 bps)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("往返交易摩擦成本 (Round-Trip Fee & Slippage in bps)", fontsize=10)
    ax1.set_ylabel("样本外 (OOS 2024-2026) 累计收益率 (%)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)

    # 2. 夏普比率敏感度曲线
    for idx, strat in enumerate(strategies):
        sub = df_res[df_res['Strategy'] == strat].sort_values('Round_Trip_Fee_bps')
        ax2.plot(sub['Round_Trip_Fee_bps'], sub['Sharpe'], label=strat, color=colors[idx], marker=markers[idx], linewidth=2.0)
        
    ax2.axhline(y=etf_sharpe, color='#000000', linestyle='--', linewidth=2.0, label=f'511380.SH ETF 基准夏普 ({etf_sharpe:.2f})')
    ax2.set_title("可转债策略夏普比率 (Sharpe Ratio) vs 往返交易摩擦费率", fontsize=12, fontweight='bold')
    ax2.set_xlabel("往返交易摩擦成本 (Round-Trip Fee & Slippage in bps)", fontsize=10)
    ax2.set_ylabel("夏普比率 (Sharpe Ratio)", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out_img = REPO_ROOT / "artifacts" / "institutional_sensitivity_chart.png"
    plt.savefig(out_img, dpi=300)
    print(f"机构敏感度分析图已保存至:\n  - {out_img}")

if __name__ == '__main__':
    main()
