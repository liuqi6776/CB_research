# -*- coding: utf-8 -*-

"""
15分钟截面打分轮动与正股驱动策略主程序（包含四阶段严谨时间切分验证）
15-Min Intraday Cross-Sectional Rotation & Stock-Lag Driving Strategy Runner
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cb_quant.data_loader import CBDataLoader
from cb_quant.intraday_cross_sectional import CBIntradayCrossSectionalEngine, CBIntradayCrossSectionalSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def evaluate_period_performance(df_eq, df_tr, stage_name):
    if df_eq.empty:
        print(f"\n【{stage_name}】无交易数据。")
        return {}
        
    df_e = df_eq.copy()
    df_e['trade_time'] = pd.to_datetime(df_e['trade_time'])
    df_e['date'] = df_e['trade_time'].dt.date
    
    daily_nav = df_e.groupby('date')['nav'].last()
    daily_ret = daily_nav.pct_change().dropna()
    
    start_val = 1000000.0
    end_val = daily_nav.iloc[-1]
    tot_ret = (end_val - start_val) / start_val
    num_days = len(daily_nav)
    ann_ret = (1.0 + tot_ret) ** (252.0 / num_days) - 1.0 if num_days > 0 else 0.0
    
    rf_daily = 0.02 / 252.0
    sharpe = ((daily_ret - rf_daily).mean() / (daily_ret.std() + 1e-8)) * np.sqrt(252.0)
    
    cummax = daily_nav.cummax()
    drawdown = (daily_nav - cummax) / cummax
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    
    sells = df_tr[df_tr['action'] != 'BUY'].copy() if not df_tr.empty else pd.DataFrame()
    total_trades = len(sells)
    win_rate = (sells['pnl'] > 0).mean() if total_trades > 0 else 0.0
    
    wins = sells[sells['pnl'] > 0]['pnl'] if total_trades > 0 else pd.Series()
    losses = abs(sells[sells['pnl'] < 0]['pnl']) if total_trades > 0 else pd.Series()
    
    avg_w = wins.mean() if len(wins) > 0 else 0.0
    avg_l = losses.mean() if len(losses) > 0 else 1.0
    profit_loss_ratio = avg_w / avg_l if avg_l > 0 else 0.0
    
    daily_t = total_trades / float(num_days) if num_days > 0 else 0.0

    print("\n" + "="*65)
    print(f"       15分钟截面轮动策略 - 【{stage_name}】 绩效评估报告")
    print("="*65)
    print(f"测试交易天数:       {num_days} 天")
    print(f"期末净资产:         {end_val:,.2f} 元")
    print(f"累计绝对收益率:     {tot_ret*100:+.2f}%")
    print(f"年化收益率:         {ann_ret*100:+.2f}%")
    print(f"夏普比率 (Sharpe):  {sharpe:.2f}")
    print(f"卡玛比率 (Calmar):  {calmar:.2f}")
    print(f"最大回撤 (MaxDD):   {max_dd*100:.2f}%")
    print(f"日均平仓交易笔数:   {daily_t:.2f} 笔/天")
    print(f"平仓交易胜率:       {win_rate*100:.2f}%")
    print(f"★ 真实盈亏比:       {profit_loss_ratio:.2f}")
    print("="*65 + "\n")
    
    return {
        'stage': stage_name,
        'tot_ret': tot_ret,
        'ann_ret': ann_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio
    }

def main():
    logging.info("=== 启动 15分钟截面打分轮动与正股驱动策略 (四阶段严谨切分验证) ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2020-01-01", end_date="2026-07-25", max_bonds=300)
    
    # 1. 交易池过滤 (价格 <= 180元，规模 >= 2.0亿，非强赎)
    logging.info("过滤交易池 (价格 <= 180, 规模 >= 2.0亿)...")
    df_filtered = CBIntradayCrossSectionalEngine.filter_universe(df_panel, max_price=180.0, min_scale=2.0)
    
    # 2. 转换为 15分钟 K 线并计算截面 Z-Score 多因子打分
    df_filtered['trade_time'] = pd.to_datetime(df_filtered['trade_time'])
    df_15m = df_filtered.groupby(['ts_code', pd.Grouper(key='trade_time', freq='15min')]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'amount': 'sum'
    }).dropna().reset_index()
    
    df_scored = CBIntradayCrossSectionalEngine.compute_cross_sectional_scores(df_15m)
    
    # 3. 四阶段切分评估
    periods = [
        ("阶段 1: 2020-2023 研究与训练", "2020-01-01", "2023-12-31"),
        ("阶段 2: 2024 参数选择",       "2024-01-01", "2024-12-31"),
        ("阶段 3: 2025 样本外验证",     "2025-01-01", "2025-12-31"),
        ("阶段 4: 2026 最终检验与实盘", "2026-01-01", "2026-07-25")
    ]
    
    simulator = CBIntradayCrossSectionalSimulator(
        initial_capital=1000000.0,
        top_n=8,
        exit_rank_pct=0.20,      # 跌出前 20% 离场
        stop_loss=-0.010,        # 单债止损 -1.0%
        max_surge_cap=0.035,     # 单根 15m 涨幅 > 3.5% 不追高
        single_slippage=0.0005,  # 单边 0.05% 滑点
        commission=0.00005
    )
    
    summary_results = []
    
    plt.figure(figsize=(14, 8))
    
    for stage_name, start_d, end_d in periods:
        mask = (df_scored['trade_time'] >= start_d) & (df_scored['trade_time'] <= end_d)
        df_sub = df_scored[mask].copy()
        
        if df_sub.empty:
            continue
            
        df_eq, df_tr = simulator.run_backtest(df_sub)
        res = evaluate_period_performance(df_eq, df_tr, stage_name)
        summary_results.append(res)
        
        if not df_eq.empty:
            plt.plot(pd.to_datetime(df_eq['trade_time']), df_eq['nav'], label=stage_name, linewidth=2.0)

    plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
    plt.title("15-Min Cross-Sectional Rotation Strategy - 4-Stage Performance Curve (2020-2026)", fontsize=14)
    plt.xlabel("Trade Time", fontsize=12)
    plt.ylabel("Net Asset Value (RMB)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig("cb_cross_sectional_equity.png", dpi=300)

if __name__ == "__main__":
    main()
