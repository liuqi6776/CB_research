# -*- coding: utf-8 -*-

"""
全量可转债 5 分钟 K 线多因子策略全历史回测 (2020 - 2026 全量 1030 只可转债)
Full Convertible Bond 5-Min Multi-Factor Backtest Engine (2020 - Present, 1030 Bonds)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cb_quant.data_loader import CBDataLoader
from cb_quant.factor_engine import CBFactorEngine
from cb_quant.scoring_engine import CBScoringEngine
from cb_quant.backtest_engine import CBMinuteBacktestEngine

# 配置日志 / Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def calculate_performance_metrics(df_equity, df_trades, initial_capital=1000000.0):
    if df_equity.empty:
        return {}
        
    nav_series = df_equity['nav']
    total_return = (nav_series.iloc[-1] - initial_capital) / initial_capital
    
    returns = nav_series.pct_change().dropna()
    ann_factor = 11616.0 # 每天 48 根 K 线 * 242 交易日
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    sharpe_ratio = (mean_ret / (std_ret + 1e-8)) * np.sqrt(ann_factor) if std_ret > 0 else 0.0
    
    cummax = nav_series.cummax()
    drawdowns = (nav_series - cummax) / cummax
    max_drawdown = drawdowns.min()
    
    num_bars = len(nav_series)
    years = num_bars / ann_factor
    ann_return = (1.0 + total_return) ** (1.0 / max(years, 0.01)) - 1.0 if total_return > -1.0 else -1.0
    calmar_ratio = ann_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0

    if not df_trades.empty and 'action' in df_trades.columns:
        sell_trades = df_trades[df_trades['action'] != 'BUY']
        total_trades = len(sell_trades)
        win_trades = len(sell_trades[sell_trades['pnl'] > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0
        
        avg_win = sell_trades[sell_trades['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0.0
        avg_loss = abs(sell_trades[sell_trades['pnl'] < 0]['pnl'].mean()) if (total_trades - win_trades) > 0 else 1.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    else:
        total_trades = 0
        win_rate = 0.0
        win_loss_ratio = 0.0

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'win_loss_ratio': win_loss_ratio,
        'final_nav': nav_series.iloc[-1]
    }

def main():
    logging.info("=== 开始运行 全量可转债 (1,030 只) 2020-至今 5分钟线多因子策略回测 ===")
    
    # 1. 加载 2020 至 2026 年全量 1,030 只可转债数据
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2020-01-01", end_date="2026-07-25", max_bonds=None)
    
    # 2. 计算分钟级多因子 (路径平滑动量、相对大盘 Alpha、30分钟振幅、量比)
    logging.info("计算全量 5分钟多因子指标矩阵...")
    df_factors = CBFactorEngine.compute_factors(df_panel)
    
    # 3. 横截面打分与 Top 8 选债
    logging.info("执行横截面 Percentile 打分与领涨池筛选...")
    scoring_engine = CBScoringEngine(top_n=8, min_amount=500000.0)
    df_scored = scoring_engine.compute_cross_sectional_scores(df_factors)
    
    # 4. 运行 T+0 事件驱动回测 (含 1.5% 动态止损与离场补位)
    logging.info("启动 T+0 分钟级事件驱动回测引擎 (单边 0.1% 滑点损耗)...")
    bt_engine = CBMinuteBacktestEngine(
        initial_capital=1000000.0,
        top_n=8,
        rank_decay_buffer=5,
        stop_loss_pct=0.015,     # 1.5% 动态止损
        single_slippage=0.001,   # 单边 0.1% 滑点
        commission=0.00005       # 万0.5 佣金
    )
    
    df_equity, df_trades = bt_engine.run_backtest(df_scored)
    
    # 5. 计算并打印全量绩效评估报告
    metrics = calculate_performance_metrics(df_equity, df_trades)
    
    print("\n" + "="*60)
    print("   全量可转债 (1,030 只) 2020-至今 5分钟线多因子策略回测报告")
    print("="*60)
    print(f"数据覆盖范围:       2020-01-01 至 2026-07-24 (全量 1,030 只个券)")
    print(f"初始资金:           1,000,000.00 元")
    print(f"期末净值:           {metrics.get('final_nav', 0.0):,.2f} 元")
    print(f"累计收益率:         {metrics.get('total_return', 0.0)*100:.2f}%")
    print(f"年化收益率:         {metrics.get('ann_return', 0.0)*100:.2f}%")
    print(f"夏普比率 (Sharpe):   {metrics.get('sharpe_ratio', 0.0):.2f}")
    print(f"卡尔玛比率 (Calmar): {metrics.get('calmar_ratio', 0.0):.2f}")
    print(f"最大回撤 (MaxDD):   {metrics.get('max_drawdown', 0.0)*100:.2f}%")
    print(f"总平仓交易次数:     {metrics.get('total_trades', 0)} 次")
    print(f"胜率 (WinRate):     {metrics.get('win_rate', 0.0)*100:.2f}%")
    print(f"盈亏比 (Profit/Loss):{metrics.get('win_loss_ratio', 0.0):.2f}")
    print("="*60 + "\n")
    
    # 6. 保存全量回测净值曲线图及交易记录
    if not df_equity.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(df_equity['trade_time']), df_equity['nav'], label='Full Market Strategy NAV (1,030 CBs)', color='#2ca02c', linewidth=1.5)
        plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
        plt.title("A-Share Full Convertible Bonds (1030 CBs) 5-Min Multi-Factor Strategy NAV (2020-2026)", fontsize=14)
        plt.xlabel("Trade Time", fontsize=12)
        plt.ylabel("Net Asset Value (RMB)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        
        output_img = "cb_full_market_strategy_equity.png"
        plt.savefig(output_img, dpi=300)
        logging.info(f"全量策略净值曲线图已生成存至: {os.path.abspath(output_img)}")
        
    if not df_trades.empty:
        df_trades.to_csv("cb_full_market_trades.csv", index=False, encoding='utf-8-sig')
        logging.info("全量交易明细日志已存至: cb_full_market_trades.csv")

if __name__ == "__main__":
    main()
